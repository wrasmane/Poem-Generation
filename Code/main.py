"""
Main Class
TODO: add info
"""
import pandas as pd

from Naive import Naive
from StateSpace import StateSpace
from Transformer import Transformer


def load_data():
    # read and save poems
    data = pd.read_csv("../Data/PoetryFoundationData.csv")
    poems = data["Poem"].tolist()

    # clean poems from leading and trailing whitespace and changes all to lowercase
    for i in range(len(poems)):
        poems[i] = poems[i].replace("\r", "")
        # poems[i] = poems[i].replace("\n\n\n", "\n")
        poems[i] = poems[i].strip()
        poems[i] = poems[i].lower()

    return poems

def user_interaction():
    model = None

    print("\nWelcome to Poem Generator!\n\n")

    print("Please choose an generation model:")
    print("Naive:\t\t'1'")
    print("Transformer:\t'2'")
    print("State Space:\t'3'\n")

    model_choice = input("Choice:\t")
    print("\n")

    if model_choice == "1":
        model = Naive()
    elif model_choice == "2":
        model = Transformer()
    elif model_choice == "3":
        model = StateSpace()

    print("Please enter a starting phrase to generate a poem from:\n")
    phrase = input("Phrase:\t")

    return model, phrase

def continue_user_interaction():
    print("\nWould you like to generate another poem with the same model?")
    generate_again = input("(y/n):\t")

    if generate_again.lower() == "y":
        print("Please enter a starting phrase to generate a poem from:\n")
        phrase = input("Phrase:\t")
        return phrase
    else:
        quit()

def main():
    poems = load_data()

    model, phrase = user_interaction()
    phrase = phrase.lower()

    model.fit(poems)

    poem = model.generate(phrase)

    print(poem)

    while True:
        phrase = continue_user_interaction()
        phrase = phrase.lower()

        poem = model.generate(phrase)

        print(poem)


if __name__ == '__main__':
    main()
