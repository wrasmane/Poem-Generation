"""
State Space Model
TODO: add info
"""

from typing import List
from Model import Model
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from sklearn.feature_extraction.text import TfidfVectorizer

MATRIX_SIZE = 10

class StateSpace(Model):
    def fit(self, data: List[str]):
        self.weight_dimensions = MATRIX_SIZE
        self.state_dimensions = MATRIX_SIZE
        self.weight_a = self.create_default_state()
        self.weight_b = self.create_default_state()
        self.weight_c = self.create_default_state()

        self.vectorizer = TfidfVectorizer()
        self.train_data = self.vectorizer.fit_transform(data)

    def generate(self, phrase):
        self.state_x = self.convert_to_vec(phrase) # Initial state is based on input phrase
        self.input_u = self.convert_to_vec(phrase) # Input U is based on purely input

        self.simulate()

    def convert_to_vec(self, string: str):
        return self.vectorizer.transform([string]).toarray()[0]

    def create_default_state(self):
        return np.zeros(shape=(self.state_dimensions, self.state_dimensions))

    def map_to_state_dimension(embedding, state_dim):
        embedding_dim = len(embedding)
        if embedding_dim > state_dim:
            # Reduce dimensionality
            return embedding[:state_dim]
        elif embedding_dim < state_dim:
            # Pad with zeros
            return np.pad(embedding, (0, state_dim - embedding_dim), mode='constant')
        return embedding

    def simulate(self):
        n_states = self.weight_dimensions
        n_outputs = self.weight_dimensions

        T = 10 # Number of steps to simulate

        observations = []

        for t in range(T):
            self.state_x = torch.add(
                torch.mul(self.weight_a, self.state_x)), torch.mul(self.weight_b, self.input_u
            )

            y = torch.mul(self.weight_c, self.state_x)
            observations.append(y)

        return observations
