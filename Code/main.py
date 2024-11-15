"""
Main Class
TODO: add info
"""
import pandas as pd
import numpy as np

from Naive import Naive
from StateSpace import StateSpace
from Transformer import Transformer


def load_data():
    # read and save poems
    data = pd.read_csv("../Data/PoetryFoundationData.csv")
    poems = data["Poem"].tolist()

    # clean poems from leading and trailing whitespace
    poems = list(map(str.strip, poems))

    return poems

def user_interaction():
    model = None
    phrase = None

    print("\nWelcome to Poem Generator!\n\n")

    print("Please choose an generation model:")
    print("Naive:\t\t'1'")
    print("Transformer:\t'2'")
    print("State Space:\t'2'\n")

    model_choice = input("Choice:\t")
    print("\n")

    if model_choice == "1":
        model = Naive
    elif model_choice == "2":
        model = Transformer
    elif model_choice == "3":
        model = StateSpace

    print("Please enter a starting phrase to generate a poem from:\n")
    phrase = input("Phrase:\t")

    return model, phrase


def main():
    poems = load_data()

    model, phrase = user_interaction()
    print(model, phrase)

if __name__ == '__main__':
    main()
