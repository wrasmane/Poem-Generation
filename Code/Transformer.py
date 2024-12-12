"""
Transformer Model

Throughout this implementation, the linked repository has been
references as a foundation and cross-reference when developing
this model
https://github.com/Suruj0001/Transfomers/tree/main

This repository worked through the paper "All you need is attention"
and provided a step-by-step walkthrough of implementing a Transformer
model from scratch. In real applications, libraries such as PyPi or
Hugging face should be used as they preform many optimizations in
calculations. Along with this, there are pretrained transformers from
Open AI that would all preform better than this one

This transformer model uses many different neural network topics as well
as classes defined in pyTorch. With this project, my main goal was to
discuss the transformer model, and not the fine details of specific aspects
such as how pytorch defines their embedding layer, etc.
"""
import math

from Model import Model
import torch
import torch.nn as nn
from torch.nn import functional
from typing import List
import tiktoken
import matplotlib.pyplot as plt

# Load model
use_trained = True
# Iterations
eval_iterations = 5 # max_iterations % eval_iterations == 0 must be true
max_iterations = 25

# Hyperparameters
size_token_embeddings = 64
dropout_rate = 0.1
context_length = 16
num_heads = 4
num_blocks = 8
device = torch.device("cpu")
batch_size = 4
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

        if use_trained:

            optim = torch.optim.Adam(self.model.parameters(), lr)
            train_losses = list()
            validation_losses = list()
            ittr = list()
            for step in range(max_iterations):
                if step % eval_iterations == 0 or step == max_iterations:
                    loss = self._loss()
                    ittr.append(step)
                    train_losses.append(round(loss["train"].item(), 3))
                    validation_losses.append(round(loss["val"].item(), 3))
                    print("Step:", step, "Training Loss:", round(loss["train"].item(), 3), "Validation Loss:",
                          round(loss["val"].item(), 3))

                x_b, y_b = self._getbatch("train")
                logs, loss = self.model(x_b, y_b)
                optim.zero_grad(True)
                loss.backward()
                optim.step()

            torch.save(self.model.state_dict(), "../Output/" + str(max_iterations) + "-model.pt")

            plt.plot(ittr, train_losses, label="Training Loss")
            plt.plot(ittr, validation_losses, label="Validation Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend(loc="best")
            plt.savefig("../Output/transformer-" + str(max_iterations) + "-loss.jpg")

        else:
            self.model.load_state_dict(torch.load("../Output/200000-model.pt"))


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
        """
        initializes this pytorch neural network module
        """
        super().__init__()
        self.size_token_embeddings = size_token_embeddings
        self.feed_forward_network = nn.Sequential(
            nn.Linear(in_features=self.size_token_embeddings, out_features=self.size_token_embeddings * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.size_token_embeddings * 4, out_features=self.size_token_embeddings),
        )

    def forward(self, x):
        """
        define how the pytorch module performs the forward pass of the network
        :param x: the abstract input data
        :return: the abstract output data
        """
        return self.feed_forward_network(x)


"""
Attention Module

gets the context a word has with other words in a phrase
"""
class Attention(nn.Module):
    def __init__(self, head_size):
        """
        initializes this pytorch neural network module
        :param head_size: the size of each attention head
        """
        super().__init__()
        self.head_size = head_size
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length

        self.key = nn.Linear(in_features=self.size_token_embeddings, out_features=self.head_size, bias=False)
        self.query = nn.Linear(in_features=self.size_token_embeddings, out_features=self.head_size, bias=False)
        self.value = nn.Linear(in_features=self.size_token_embeddings, out_features=self.head_size, bias=False)

        #used for masking
        self.register_buffer("tril", torch.tril(torch.ones((self.context_length, self.context_length))))

    def forward(self, x):
        """
        define how the pytorch module performs the forward pass of the network
        :param x: the abstract input data
        :return: the abstract output data
        """
        batch_size, time_steps, dimensions = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = (q @ k.transpose(-2, -1)) # matmul
        weights = weights * (1.0 / math.sqrt(k.size(-1))) # scale
        weights = weights.masked_fill(
            self.tril[:time_steps, :time_steps] == 0, -float("inf")) # mask
        weights = functional.softmax(weights, dim=-1) # softmax
        weights = weights @ v # matmul

        return weights

"""
Multi-Headed Attention Module

multiple attention modules used to get better insight on the phrases and context
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        """
        initializes this pytorch neural network module
        :param head_size: the size of each attention head
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length

        self.heads = nn.ModuleList([Attention(head_size) for _ in range(self.num_heads)])
        self.protection = nn.Linear(in_features=self.size_token_embeddings, out_features=self.size_token_embeddings)

    def forward(self, x):
        """
        define how the pytorch module performs the forward pass of the network
        :param x: the abstract input data
        :return: the abstract output data
        """
        out = torch.cat([attention(x) for attention in self.heads], dim=-1)
        out = self.protection(out)
        return out

"""
Main transformer block of the transformer

combines the attention modules and feedforward network
"""
class TransformerBlock(nn.Module):
    def __init__(self, num_heads):
        """
        initializes this pytorch neural network module
        :param num_heads: the number of heads to include for
        the multi-head attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length
        self.head_size = size_token_embeddings // num_heads

        self.multihead_attention = MultiHeadAttention(self.head_size)
        self.ff = FeedForward()
        self.norm1 = nn.LayerNorm(self.size_token_embeddings)
        self.norm2 = nn.LayerNorm(self.size_token_embeddings)

    def forward(self, x):
        """
        defines how the pytorch module performs the forward pass of the network
        :param x: the input data of the positional encoded word embeddings
        :return: the abstract output data
        """
        x = x + self.multihead_attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

"""
Main LLM

combines the transformer blocks with positional encoding/decoding of token embeddings
"""
class TransformerLLM(nn.Module):
    def __init__(self, max_token_value):
        """
        initializes this pytorch neural network module
        :param max_token_value: the maximum value of the tokenized text
        """
        super().__init__()
        self.size_token_embeddings = size_token_embeddings
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_token_value = max_token_value

        self.lookup_table = nn.Embedding(
            self.max_token_value + 1,
            self.size_token_embeddings
        )

        self.blocks = nn.Sequential(*(
            [TransformerBlock(num_heads) for _ in range(self.num_blocks)] +
            [nn.LayerNorm(self.size_token_embeddings)]
        ))

        self.output = nn.Linear(self.size_token_embeddings, self.max_token_value)

    def forward(self, x, targets=None):
        """
        defines how the pytorch module performs the forward pass of the network
        :param x: the input data of tokenized text
        :param targets: the true output (used for calculating loss)
        :return: the logit output and the loss (if targets is defined)
        """
        B, T = x.shape
        position_embedding = torch.zeros(self.context_length, self.size_token_embeddings)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.size_token_embeddings, 2).float() *
                             (-math.log(10000.0) / self.size_token_embeddings))

        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding[:T, :]

        x = self.lookup_table(x) + position_embedding
        x = self.blocks(x)
        logs = self.output(x)

        if targets is not None:
            B, T, C = logs.shape
            logs = logs.view(B * T, C)
            targets = targets.view(B * T)
            loss = functional.cross_entropy(logs, targets)
        else:
            loss = None

        return logs, loss

    def gen(self, x, max_new_tokens):
        """
        generates a poem from the trained model
        :param x: the tokenized input phrase
        :param max_new_tokens: the maximum number of new tokens that
        we can generate
        :return: the poem generated from the trained model in a tokenized
        text
        """
        for _ in range(max_new_tokens):
            logs, loss = self(x[:, -self.context_length:])
            previous_log = logs[:, -1, :]
            probs = functional.softmax(previous_log, dim=-1)
            next_x = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_x), dim=1)
        return x