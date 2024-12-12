# Poem Generation

## Contributors
Ethan Wrasman\
Jack Rosenbecker

## Overview
### Problem
The problem we aimed to solve in this final project was with text generation.
Generative AI is the future as seen with ChatGPT and many other Large Language
Models, and we aim to analyze and get a deeper understanding on different 
implementations. 
### Solution
The two main implementations of LLMs that we are focusing on are the Transformer
model and the State Space model. Along with this, we implemented a Naive model
that does not do poem generation, and instead pulls a poem from the dataset that
is the most similar to the input.
### Results
Our two models show promising results given the time and power we had to implement
and train them. However, it should be noted that our implementations would never
be used in the real world as many libraries exist already for them, as well as
pretrained models that already have some basic understanding of words and how
they relate with others. Our model are completely from scratch and were expected
to struggle.

## Repository Breakdown
### Code
This folder contains all the code from our project.
#### main.py
This file contains the main executable for our project.
#### Model.py
This file defines an interface for each of our models to implement.
#### Naive.py
This file contains the naive model to generate a poem
#### StateSpace.py
This file contains the state space model to generate a poem
#### Transformer.py
This file contains the transformer model to generate a poem
### Data
This folder contains the data our models need to use
#### PoetryFoundationData.csv
This file contains the dataset from [Kaggle](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems)
### Output
This folder contains any output generated from the models
#### 200000-model.pt
This file contains a saved state of the transformer model
#### there-once-was-a-dog-Naive.txt
This file contains the gnerated output form the naive model with the input
phrase of "there once was a dog"
#### there-once-was-a-dog-Transformer.txt
This file contains the generated output from the transformer model with the
input phrase of "there once was a dog"
#### transformer-200000-loss.jpg
This file contains the saved plot from the training of the transformer model
with 200000 iterations
#### TODO WITH STATE SPACE MODEL
TODO WITH STATE SPACE MODEL
### README.md
This file contains the readme for this project

## Running the Code
The following installations require the use of the pip package manager.
### Main
There is only one additional library defined in the main class that will need
to be installed. Pandas can be installed by running the command `pip install pandas`\
\
The main file saves the generated poems to the Output directory. The output file 
named {input phrase}-{model name}.txt. If there is an error saving the poem, the content
of the file will contain "Error saving poem"
### State Space
TODO Installs\
\
TODO saved files
### Transformer
There are few different libraries that will need to be installed in order to 
run the transformer model. Pytorch can be installed by running the command
`pip install torch`. Tiktoken can be installed by running the command `pip install tiktoken`.
Lastly, matplotlib can be installed by running the command `pip install matplotlib`\
\
This file produces a few outputs that save to the Output directory depending on one main 
attribute. First if the boolean `used_trained`(located in Transformer.py) is true, there 
will be no outputs from this file. If `used_trained` is false, then the file will create 
a new transformer model with the given hyperparameters. `eval_iterations` and `max_iterations` 
are the main tunable parameters, where the default values for training time sake is 5 and 
25 respectively. With this setup, two new files will be produced as a result. `25-model.pt` 
will be the saved state of the transformer model and `transformer-25-loss.jpg` will contain 
the saved loss plot from the validation cycles.
