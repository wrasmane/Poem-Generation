"""
Transformer Model

Code was adopted and modeled after the below repository
https://github.com/Suruj0001/Transfomers/tree/main

This repository worked through the paper "All you need is attention"
and provided a step-by-step walkthrough of implementing a Transformer
model from scratch. In real applications, libraries such as PyPi or
Hugging face. Along with this, there are pretrained transformers from
Open AI that would all preform better than this one

This transformer model uses many advanced neural network topics that
are above the level for this class, so the main components were
discussed in class, such as the attention module and the positional
encoding/decoding.

"""
import math

from Model import Model
import torch
import torch.nn as nn
from torch.nn import functional
from typing import List
import tiktoken
import matplotlib.pyplot as plt


# Hyperparameters
size_token_embeddings = 64
dropout_rate = 0.1
context_length = 16
num_heads = 4
num_blocks = 8
device = torch.device("cpu")
batch_size = 4
eval_iterations = 1
max_iterations = 10
lr = 0.0001
max_new_tokens = 100

"""
Main transformer class

"""
class Transformer(Model):
    def fit(self, data: List[str]):
        """
        joins all poems into one big string
        tokenizes them usng tiktoken
        initializes the LLM
        trains using the adam optimizer

        :param data: poem data in a list of strings
        """
        self.poems = "\n\n".join(data)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        tokenized_text = self.encoding.encode(self.poems)
        self.max_token_value = max(tokenized_text) + 1                                  # the maximum value of the tokenized numbers
        tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # put tokenized text into tensor

        split = int(len(tokenized_text) * 0.8)
        self.train = tokenized_text[:split]
        self.val = tokenized_text[split:]

        self.model = TransformerLLM(self.max_token_value)
        self.model = self.model.to(device)

        optim = torch.optim.Adam(self.model.parameters(), lr)
        train_losses = list()
        validation_losses = list()
        for step in range(max_iterations):
            if step % eval_iterations == 0 or step == max_iterations:
                loss = self._loss()
                train_losses.append(round(loss["train"].item(), 3))
                validation_losses.append(round(loss["val"].item(), 3))
                print("Step:", step, "Training Loss:", round(loss["train"].item(), 3), "Validation Loss:",
                      round(loss["val"].item(), 3))

            x_b, y_b = self._getbatch("train")
            logs, loss = self.model(x_b, y_b)
            optim.zero_grad(True)
            loss.backward()
            optim.step()

        torch.save(self.model.state_dict(), str(max_iterations) + "-model.pt")

        plt.plot(range(max_iterations), train_losses, label="Training Loss")
        plt.plot(range(max_iterations), validation_losses, label="Validation Loss")
        plt.legend(loc="best")
        plt.savefig("transformer-" + str(max_iterations) + "-loss.jpg")


    def _getbatch(self, split):
        """
        helper method for getting a batch of input data
        :param split: either the training for validation split for labeling the data
        :return: data for input and data + 1 token as output
        """
        data = self.train if split == "train" else self.val
        idx = torch.randint(0, len(data) - context_length, (batch_size,))
        x = torch.stack([data[i:i + context_length] for i in idx]).to(device)
        y = torch.stack([data[i+1:i + context_length+1] for i in idx]).to(device)
        return x, y

    @torch.no_grad()
    def _loss(self):
        """
        calculates the loss of the model while training
        used to minimize in the adam optimizer
        :return: loss per eval iteration
        """
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iterations)
            for k in range(eval_iterations):
                x_b, y_b = self._getbatch(split)
                logs, loss = self.model(x_b, y_b)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def generate(self, phrase: str) -> str:
        """
        generates the output of the model when given an input phrase
        uses a max_new_token int for the maximum number of tokesn generated
        :param phrase: input phrase from the user
        :return: the generated poem
        """
        self.model.eval()
        start_tokens = self.encoding.encode(phrase)
        x = (torch.tensor(start_tokens, device=device)[None, ...])
        y = self.model.gen(x, max_new_tokens)
        return self.encoding.decode(y[0].tolist())


"""
Feed Forward Network

adds layers for nn understanding of the data
"""
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.size_token_embeddings = size_token_embeddings
        self.dropout_rate = dropout_rate
        self.feed_forward_network = nn.Sequential(
            nn.Linear(in_features=self.size_token_embeddings, out_features=self.size_token_embeddings * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.size_token_embeddings * 4, out_features=self.size_token_embeddings),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.feed_forward_network(x)


"""
Attention Module

gets the context a word has with other words in a phrase
"""
class Attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length
        self.dropout_rate = dropout_rate

        self.key = nn.Linear(in_features=self.size_token_embeddings, out_features=self.head_size, bias=False)
        self.query = nn.Linear(in_features=self.size_token_embeddings, out_features=self.head_size, bias=False)
        self.value = nn.Linear(in_features=self.size_token_embeddings, out_features=self.head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((self.context_length, self.context_length))))
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        batch_size, time_steps, dimensions = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, -float("inf"))

        weights = functional.softmax(weights, dim=-1)

        weights = self.dropout(weights)

        return weights @ v

"""
Multi-Headed Attention Module

multiple attention modules used to get better insight on the phrases and context
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length
        self.dropout_rate = dropout_rate

        self.heads = nn.ModuleList([Attention(head_size) for _ in range(self.num_heads)])
        self.protection = nn.Linear(in_features=self.size_token_embeddings, out_features=self.size_token_embeddings)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        out = torch.cat([attention(x) for attention in self.heads], dim=-1)
        out = self.protection(out)
        out = self.dropout(out)
        return out

"""
Main transformer block of the transformer

combines the attention modules and feedforward network
"""
class TransformerBlock(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length
        self.dropout_rate = dropout_rate
        self.head_size = size_token_embeddings // num_heads

        self.multihead_attention = MultiHeadAttention(self.head_size)
        self.ff = FeedForward()
        self.norm1 = nn.LayerNorm(self.size_token_embeddings)
        self.norm2 = nn.LayerNorm(self.size_token_embeddings)

    def forward(self, x):
        x = x + self.multihead_attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

"""
Main LLM

combines the transformer blocks with positional encoding/decoding of token embeddings
"""
class TransformerLLM(nn.Module):
    def __init__(self, max_token_value):
        super().__init__()
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.max_token_value = max_token_value
        self.lookup_table = nn.Embedding(self.max_token_value + 1, self.size_token_embeddings)

        self.blocks = nn.Sequential(*(
            [TransformerBlock(num_heads) for _ in range(self.num_blocks)] +
            [nn.LayerNorm(self.size_token_embeddings)]
        ))
        self.output = nn.Linear(self.size_token_embeddings, self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        lookup_table = torch.zeros(self.context_length, self.size_token_embeddings)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.size_token_embeddings, 2).float() * (-math.log(10000.0) / self.size_token_embeddings))
        lookup_table[:, 0::2] = torch.sin(position * div_term)
        lookup_table[:, 1::2] = torch.cos(position * div_term)
        position_embedding = lookup_table[:T, :]
        x = self.lookup_table(idx) + position_embedding
        x = self.blocks(x)
        logs = self.output(x)

        if targets is not None:
            B, T, C = logs.shape
            logs_reshaped = logs.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = functional.cross_entropy(logs_reshaped, targets_reshaped)
        else:
            loss = None

        return logs, loss

    def gen(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logs, loss = self(idx[:, -self.context_length:])
            last_log = logs[:, -1, :]
            probs = functional.softmax(last_log, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx