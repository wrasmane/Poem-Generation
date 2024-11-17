"""
Transformer Model
TODO: add info
"""
import math

from Model import Model
import torch
import torch.nn as nn
from torch.nn import functional

# Hyperparameters
num_token_embeddings = 64
dropout_rate = 0.1
context_length = 16
num_heads = 0

class Transformer(Model):
    def fit(self):
        pass

    def generate(self, phrase):
        pass

# Feed Forward network
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_token_embeddings = num_token_embeddings
        self.dropout_rate = dropout_rate
        self.feed_forward_network = nn.Sequential(
            nn.Linear(in_features=self.num_token_embeddings, out_features=self.num_token_embeddings * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.num_token_embeddings * 4, out_features=self.num_token_embeddings),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.feed_forward_network(x)


# Attention
class Attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.num_token_embeddings = num_token_embeddings
        self.context_length = context_length
        self.dropout_rate = dropout_rate

        self.key = nn.Linear(in_features=self.num_token_embeddings, out_features=self.self.head_size, bias=False)
        self.query = nn.Linear(in_features=self.num_token_embeddings, out_features=self.self.head_size, bias=False)
        self.value = nn.Linear(in_features=self.num_token_embeddings, out_features=self.self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((self.context_length, self.context_length))))
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        batch_size, time_steps, dimensions = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, -float('inf'))

        weights = functional.softmax(weights, dim=-1)

        weights = self.dropout(weights)

        return weights @ v

# Multi-headed Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_token_embeddings = num_token_embeddings
        self.context_length = context_length
        self.dropout_rate = dropout_rate

        self.heads = nn.ModuleList([Attention(head_size) for _ in range(self.num_heads)])
        self.protection = nn.Linear(in_features=self.num_token_embeddings, out_features=self.num_token_embeddings)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        out = torch.cat([attention(x) for attention in self.heads], dim=-1)
        out = self.protection(out)
        out = self.dropout(out)
        return out
