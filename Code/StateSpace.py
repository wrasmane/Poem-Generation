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

MATRIX_SIZE = 10

class StateSpace(Model):
    def fit(self, data: List[str]):
        self.weight_dimensions = MATRIX_SIZE
        self.state_dimensions = MATRIX_SIZE
        self.weight_a = self.create_default_matrix()
        self.weight_b = self.create_default_matrix()
        self.weight_c = self.create_default_matrix()
        self.state_x = self.create_default_matrix()
        pass

    def generate(self, phrase):
        pass

    def create_default_matrix(self):
        return torch.randn(self.weight_dimensions, self.weight_dimensions);

    def create_default_state(self):
        return np.zeros(shape=(self.state_dimensions, self.state_dimensions))

    def next_state(self, input: str):
        n_states = self.weight_dimensions
        n_outputs = self.weight_dimensions

        T = len(input) # Number of time steps, one for each char

        observations = []

        for t in range(T):
            token = input[t]
            
            self.state_x = torch.add(
                torch.mul(self.weight_a, self.state_x)), torch.mul(self.weight_b, token
            )

            y = torch.mul(self.weight_c, self.state_x)
            observations[y] = y