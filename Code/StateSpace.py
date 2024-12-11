"""
State Space Model
TODO: add info
"""
from typing import List

from typing import List

from matplotlib import pyplot as plt
from Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
load_path = ""

STATE_DIM = 64
EMBED_DIM = 32

epochs = 2

class StateSpace(Model):
    def fit(self, data: List[str]):
        if load_path != "":
            self.model.load_state_dict(torch.load(load_path))
            return

        print(f"Using inference device for SSM:  {device}")
        # poem data tokenization
        self.poems = "\n\n".join(data)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        tokenized_text = self.tokenizer.encode(self.poems)
        self.text_tensor = torch.tensor(tokenized_text, dtype=torch.long, device=device) 
        self.vocab_size = self.tokenizer.n_vocab

        seq_len = 5
        dataset = StringDataset(self.poems, seq_len, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.model = TextStateSpaceModel(self.vocab_size, EMBED_DIM, STATE_DIM).to(device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        train_losses = list()
        ittr = list()

        for epoch in range(epochs):
            print(f"epoch: {epoch}")
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                batch_size = inputs.size(0)
                x_t = torch.zeros(batch_size, STATE_DIM, device=device)  # Initial state
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

                if batch_idx % 1000 == 0:  # Log every 10 batches
                    print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

                ittr.append(epoch)
                train_losses.append(round(loss.item(), 3))

        torch.save(self.model.state_dict(), "../Output/" + str(epochs) + "-ss-model.pt")

        plt.plot(ittr, train_losses, label="Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.savefig("../Output/ssm-" + str(epoch) + "-loss.jpg")

    def generate(self, phrase: str) -> str:
        seed_tokens = self.tokenizer.encode(phrase)
        x_t = torch.zeros(1, STATE_DIM, device=device)  # Add batch dimension
        generated_text = phrase
        current_tokens = seed_tokens

        length = 200
        temperature = 0.8

        for _ in range(length):
            # Process last token
            u_t = self.model.embedding(torch.tensor(current_tokens[-1], device=device)).unsqueeze(0)
            x_t, y_t = self.model(x_t, u_t)
            
            # Apply temperature to logits for more diverse sampling
            logits = y_t @ self.model.embedding.weight.T
            logits = logits / temperature
            probabilities = F.softmax(logits, dim=-1)
            
            # Sample from probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            
            generated_text += self.tokenizer.decode([next_token])
            current_tokens.append(next_token)

        return generated_text


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
        # Reshape x_t to match (state_dim, batch_size) for compatibility
        x_t = x_t.unsqueeze(-1)  # (batch_size, state_dim) -> (batch_size, state_dim, 1)
        u_t = u_t.unsqueeze(-1)  # (batch_size, embed_dim) -> (batch_size, embed_dim, 1)

        # State transition
        x_next = torch.matmul(self.A, x_t).squeeze(-1) + torch.matmul(self.B, u_t).squeeze(-1) + self.bias_state

        # Observation
        y_t = torch.matmul(self.C, x_next.unsqueeze(-1)).squeeze(-1) + self.bias_obs

        return x_next, y_t

PERCENT = 0.25 # Use only 25% of our data

class StringDataset(Dataset):
    def __init__(self, text, seq_len, tokenizer):
        self.seq_len = seq_len
        self.text = tokenizer.encode(text)  # Tokenize text into token IDs
        reduced_length = int(len(self.text) * PERCENT)
        self.text = self.text[:reduced_length]
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
