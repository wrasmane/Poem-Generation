"""
State Space Model
TODO: add info
"""

from typing import List
from Model import Model
import torch
import torch.nn as nn
from torch.nn import functional

MATRIX_SIZE = 10

class StateSpace(Model):
    def fit(self, data: List[str]):
        self.weight_dimensions = MATRIX_SIZE
        self.weight_a = self.create_default_matrix()
        self.weight_b = self.create_default_matrix()
        self.state = self.create_default_matrix()
        pass

    def generate(self, phrase):
        pass

    def create_default_matrix(self):
        return nn.Parameter(torch.randn(self.weight_dimensions, self.weight_dimensions));