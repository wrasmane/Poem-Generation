"""
Model Interface
TODO: add info
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import List

class Model:
    def fit(self, data: List[str]):
        pass

    def generate(self, phrase: str) -> str:
        pass
