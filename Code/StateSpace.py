"""
State Space based poem generation model. 
Uses the format of 
x = Ax(t) + Bu(t)
y = Cx(t) + Du(t)

The model will also automatically use a CUDA device via pytorch if one is available.
If there are no available CUDA devices it will default to CPU training and inference.
"""

from typing import List
from Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
load_path = "../Output/2-ss-model.pt"
# load_path = ""

STATE_DIM = 128
EMBED_DIM = 64
epochs = 1

"""
High level State Space Model implementation.
This acts as a shim between the main functionality of our testing program and
the model itself.
"""
class StateSpace(Model):
    """
    Trains the model based on the data given.
    This make take quite a while depending on your hardware.
    If there is a model provided via the "load_path" variable
    it will use that instead.
    """
    def fit(self, data: List[str]):
        device_name = ""

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = device

        print(f"Using inference device for SSM:  {device_name}")

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

        if load_path != "":
            self.model.load_state_dict(torch.load(load_path))
            return

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

        torch.save(self.model.state_dict(), "../Output/" + str(epochs) + "-ss-model.pt")

    """
    Generates a poem based on the given phrase input.
    """
    def generate(self, phrase: str) -> str:
        seed_tokens = self.tokenizer.encode(phrase)
        x_t = torch.zeros(STATE_DIM, device=device)
        generated_text = phrase

        LENGTH = 200 # Generate 200 tokens

        for _ in range(LENGTH):
            for token in seed_tokens:
                u_t = self.model.embedding(torch.tensor(token, device=device))
                x_t, y_t = self.model(x_t, u_t)

            logits = y_t @ self.model.embedding.weight.T
            probabilities = F.softmax(logits, dim=0)
            next_token = torch.multinomial(probabilities, 1).item()
            
            generated_text += self.tokenizer.decode([next_token])
            seed_tokens = [next_token]  # Update seed for the next step

        return generated_text

"""
Text abstraction over state space model.
Implement's pytorch model object for use with torch.
"""
class TextStateSpaceModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, state_dim):
        super(TextStateSpaceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, embed_dim))
        self.C = nn.Parameter(torch.randn(embed_dim, state_dim))

        self.A = torch.nn.init.xavier_uniform_(self.A)
        self.B = torch.nn.init.xavier_uniform_(self.B)
        self.C = torch.nn.init.xavier_uniform_(self.C)

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

PERCENT = 0.3 # Use only 30% of our data

"""
Dataset object containing all of our poem data.
This allows for much easier training as the chunking during training is handled by torch.
"""
class StringDataset(Dataset):
    def __init__(self, text, seq_len, tokenizer):
        self.seq_len = seq_len
        self.text = tokenizer.encode(text)  # Tokenize text into token IDs
        reduced_length = int(len(self.text) * PERCENT)
        self.text = self.text[:reduced_length]
        self.inputs, self.targets = self.create_sequences()
    
    """
    Creates input and output data points for training.
    """
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