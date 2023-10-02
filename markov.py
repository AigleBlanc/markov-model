import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
import seaborn as sn

from google.colab import drive
drive.mount('/content/drive')

#load the training data (.txt file) from your drive
url = 'https://drive.google.com/uc?id={}'.format("fileID_from drive")
df = pd.read_csv(url, delimiter= "\t", header = None, names = ["text"])

texts = np.array(df["text"])
text = ' '.join(texts)
print(text.split())

input_data = text

#training
import random
from collections import defaultdict

def train_markov_chain(text, k):
    words = text.split()
    states = [tuple((words + words[0:k-1])[i:i+k]) for i in range(len(words))]

    neighbors_dict = defaultdict(list)
    for state in states:
      neighbors_dict[state[:-1]].append(state)

    return neighbors_dict

training_data = input_data
k_val = 3
neighbors_dictionary = train_markov_chain(training_data, k_val)

#execution

def execute_markov_chain(start, num_words, neighbors_dict):
  initial_state = tuple(start.split())

  def next(v):
    neighbors = neighbors_dict[v[1:]]
    return random.choice(neighbors)

  def generate_text():
    current_state = initial_state
    generated_text = list(current_state)

    # Generate text until the required number of words is reached
    while current_state[1:] in neighbors_dict and len(generated_text) < num_words:
      next_state = next(current_state)
      generated_text.append(next_state[-1])
      current_state = next_state

    return ' '.join(generated_text)

  return generate_text()

starting_phrase = input("Enter a starting phrase/input (of 3 words): ")
num_words_to_generate = int(input("Enter the number of words to generate: "))

output_text = execute_markov_chain(starting_phrase, num_words_to_generate, neighbors_dictionary)

if tuple(starting_phrase.split()[1: ]) in neighbors_dictionary:
  print(output_text)
else:
  print('I am sorry, I do not recognize this phrase. As a model trained on a specific data, my knowledge is limited to the training data')
