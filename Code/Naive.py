"""
Naive Model
TODO: add info
"""

from typing import List
from Model import Model
import difflib


class Naive(Model):
    def fit(self, data: List[str]):
        self.poems = data
        self.min_similarity = 0.05 # Minimum 50% similarity
        pass

    def generate(self, phrase: str) -> str:
        similar_poem = ""
        similar_score = -1

        for poem in self.poems:
            ratio = difflib.SequenceMatcher(None, poem, phrase).ratio()

            # Find the most similar poem
            if ratio > similar_score:
                similar_score = ratio
                similar_poem = poem
        
        if similar_score > self.min_similarity:
            return similar_poem
        return "" # Failure