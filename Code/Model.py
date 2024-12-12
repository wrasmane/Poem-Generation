"""
Model Interface

this class contains the main interface for creating a
model. In includes a fit and a generate function that
are used when calling the model in the main file
"""

from typing import List

class Model:
    def fit(self, data: List[str]):
        """
        this method will start the training of the model.
        :param data: the poems the model intends to learn
        """
        pass

    def generate(self, phrase: str) -> str:
        """
        this method will generate a poem from the given phrase.
        :param phrase: the input phrase to generate a poem from
        :return: the generated poem
        """
        pass