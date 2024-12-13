"""
Naive Model

this class contains the implementation of the naive model.
Instead of only comparing if the input phrase exists in the
dataset, we took advise to use similarity where the generated
poem will be whichever one is the most similar to the input
phrase.
"""

from typing import List
from Model import Model
import difflib


class Naive(Model):
    def fit(self, data: List[str]):
        """
        the training phase for the naive model is simply
        storing the poems to use later and initialize a
        hyperparameter
        :param data: the poem data
        """
        self.poems = data
        self.min_similarity = 0.05 # Minimum 50% similarity
        pass

    def generate(self, phrase: str) -> str:
        """
        the generation of a poem using the naive model compares
        the similarity between an input phrase and any existing
        poem in the dataset. The returned poem will be one that
        is the most similar to the input phrase
        :param phrase: the input phrase for the poem
        :return: the most similar poem that exists in the dataset
        """
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