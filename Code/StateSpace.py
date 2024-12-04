"""
State Space Model
TODO: add info
"""
from typing import List

from typing import List
from Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken

STATE_DIM = 32
EMBED_DIM = 16

class StateSpace(Model):
    def fit(self, data: List[str]):
        # poem data tokenization
        self.poems = "\n\n".join(data)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        tokenized_text = self.encoding.encode(self.poems)
        self.text_tensor = torch.tensor(tokenized_text, dtype=torch.long, device=torch.device) 
        self.vocab_size = self.tokenizer.n_vocab

        seq_len = 5
        dataset = StringDataset(self.poems, seq_len)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.model = TextStateSpaceModel(self.vocab_size, EMBED_DIM, STATE_DIM)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        epochs = 10
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                batch_size = inputs.size(0)
                x_t = torch.zeros(batch_size, STATE_DIM)  # Initial state
                loss = 0
                
                for t in range(seq_len):
                    u_t = self.model.embedding(inputs[:, t])
                    x_t, y_t = self.model(x_t, u_t)
                
                # Predict next token
                logits = y_t @ self.model.embedding.weight.T  # Project back to vocab space
                loss = criterion(logits, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            

    def generate(self, phrase: str) -> str:
        seed_tokens = self.tokenizer.encode(phrase)
        x_t = torch.zeros(STATE_DIM)
        generated_text = phrase

        LENGTH = 50 # Generate 50 tokens

        for _ in range(LENGTH):
            for token in seed_tokens:
                u_t = self.model.embedding(torch.tensor(token))
                x_t, y_t = self.model(x_t, u_t)
            logits = y_t @ self.model.embedding.weight.T
            next_token = torch.argmax(F.softmax(logits, dim=0)).item()
            generated_text += self.tokenizer.decode([next_token])
            seed_tokens = [next_token]  # Update seed for the next step

        return generated_text

    def convert_to_tensor(self, string: str):
        token = self.encoding.encode(string)
        return torch.tensor(token, dtype=torch.long)


class TextStateSpaceModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, state_dim):
        super(TextStateSpaceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, embed_dim))
        self.C = nn.Parameter(torch.randn(embed_dim, state_dim))
        self.bias_state = nn.Parameter(torch.zeros(state_dim))
        self.bias_obs = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, x_t, u_t):
        x_next = torch.matmul(self.A, x_t) + torch.matmul(self.B, u_t) + self.bias_state
        y_t = torch.matmul(self.C, x_next) + self.bias_obs
        return x_next, y_t
    
class StringDataset(Dataset):
    def __init__(self, text, seq_len, tokenizer):
        self.seq_len = seq_len
        self.text = tokenizer.encode(text)  # Tokenize text into token IDs
        self.inputs, self.targets = self.create_sequences()
    
    def create_sequences(self):
        inputs, targets = [], []
        for i in range(len(self.text) - self.seq_len):
            inputs.append(self.text[i:i + self.seq_len])
            targets.append(self.text[i + self.seq_len])
        return torch.tensor(inputs), torch.tensor(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
