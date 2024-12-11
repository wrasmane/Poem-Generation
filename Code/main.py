"""
Main Class

This class runs the main workflow of the poem generator.

The data is loaded from the poetry foundation dataset and
preprocessed. The user picks a model from the user interaction
and loads it into the poem generator with the fit method. When
fit is called, the model will then begin its training phase.
Once the training phase is complete, the poem generator will
generate a poem from the given user input. Along with this
an output will be produced showing the training time and the
generation time for the model chosen. After the poem is outputted,
it is then saved to a file under the Output directory. Depending
on what characters generated, the file may fail to save the
content of the poem. The user will then be asked if they wanted
to enter any more poems into the same model.
"""

# imports
import time
import pandas as pd
from Naive import Naive
from StateSpace import StateSpace
from Transformer import Transformer


def load_data():
    """
    this method load the poems from the poetry foundation dataset
    and does some basic preprocessing with removing un-needed
    data, stripping leading and trailing white space, removing
    carriage return characters (for output mainly), and making
    all words be lowercase
    :return: a list of the preprocessed poems
    """
    # load from dataset
    data = pd.read_csv("../Data/PoetryFoundationData.csv")
    poems = data["Poem"].tolist()

    # clean poems from leading and trailing whitespace and changes all to lowercase
    for i in range(len(poems)):
        poems[i] = poems[i].replace("\r", "")
        poems[i] = poems[i].strip()
        poems[i] = poems[i].lower()

    return poems

def pick_model():
    """
    this method contains the main user interaction that
    picks a model to use and the first input phrase
    :return: the chosen model
    """
    model = None

    print("\nWelcome to Poem Generator!\n\n")

    print("Please choose an generation model:")
    print("Naive:\t\t\t'1'")
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

    return model

def pick_phrase():
    """
    this function picks the starting phrase to generate a
    poem from
    :return: the starting phrase
    """
    print("Please enter a starting phrase to generate a poem from:\n")
    phrase = input("Phrase:\t")
    phrase = phrase.lower()

    return phrase

def continue_user_interaction():
    """
    this function asks if the user wants to continue generating
    a poem
    :return: the phrase if the user continues, otherwise the
    application ends
    """
    print("\nWould you like to generate another poem with the same model?")
    generate_again = input("(y/n):\t")

    if generate_again.lower() == "y":
        return pick_phrase()
    else:
        quit()

def save_poem(phrase, model, poem):
    """
    this function saves the poem to a file. If the poem
    cannot be written to the file, an error message is
    displayed and the file is saved with the following
    message: "Error saving poem"
    :param phrase: the input phrase for the generation
    :param model: the model that generated the poem
    :param poem: the poem to be saved
    """
    name = ""
    if type(model) == type(Naive()):
        name = "Naive"
    elif type(model) == type(Transformer()):
        name = "Transformer"
    else:
        name = "StateSpace"

    try:
        with open("../Output/" + phrase.replace(" ", "-") + "-" + name + ".txt", "w") as file:
            file.write(poem)
    except UnicodeEncodeError:
        with open("../Output/" + phrase.replace(" ", "-") + "-" + name + ".txt", "w") as file:
            file.write("Error saving poem")
        print("\nError saving poem.")


def main():
    """
    the main function that runs the application
    """
    poems = load_data()

    model = pick_model()

    start = time.time()
    model.fit(poems)
    end = time.time()

    print("Training Time: ", end - start, "seconds")

    phrase = pick_phrase()

    start = time.time()
    poem = model.generate(phrase)
    end = time.time()

    print("Generation Time: ", end - start, "seconds")

    print(poem)

    save_poem(phrase, model, poem)

    while True:
        phrase = continue_user_interaction()

        start = time.time()
        poem = model.generate(phrase)
        end = time.time()

        print("Generation Time: ", end - start, "seconds")

        print(poem)

        save_poem(phrase, model, poem)


if __name__ == '__main__':
    main()
