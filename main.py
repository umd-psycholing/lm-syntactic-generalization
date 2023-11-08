from typing import Iterable
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from nltk.grammar import CFG
from nltk.parse import generate

import generate_corpora
import surprisal
import grammars


# arser = argparse.ArgumentParser(description="argparse")
# arser.add_argument('john', help='this is a john!')
# rgs = parser.parse_args()


def build_figure_from_data(surprisal_effects: Iterable[float], plot_title: str, color: str = 'blue'):
    num_effects = len(surprisal_effects)
    x_values = np.arange(num_effects)
    num_effects_greater_than_zero = sum(
        1 for surprisal_effect in surprisal_effects if surprisal_effect > 0)
    plt.scatter(x_values, surprisal_effects, s=2, alpha=0.5, color=color)
    plt.xlabel(
        f'{round(num_effects_greater_than_zero / num_effects, 2)} > 0 ({num_effects_greater_than_zero} / {num_effects})')
    plt.ylabel('Δ-filler - Δ+filler')
    plt.title(plot_title)
    plt.axhline(linestyle='-', label='y=0', color='black', alpha=0.75)
    # plt.show()


"""
# generate all atb sentence tuples
atb_training = []
atb_testing = []
for i, text_grammar in enumerate(grammars.ATB_GRAMMARS):
    grammar = CFG.fromstring(text_grammar)
    training, testing = generate_corpora.generate_train_test_sentence_tuples_from_grammar(
        grammar=grammar,
        split_ratio=0.65)

    atb_training.extend(training)
    atb_testing.extend(testing)

    print(f"Training, testing sets generated for ATB Grammar {i + 1}.")

# calculate surprisal on atb training set
atb_training_surprisals = []
for i, atb_tuple in enumerate(atb_training):
    atb_training_surprisals.append(
        surprisal.surprisal_effect_full_tuple(atb_tuple, "gpt2", True))

# plot data
plt.figure(1)
build_figure_from_data(surprisal_effects=atb_training_surprisals,
                       plot_title="ATB Surprisals: GPT2 | Training",
                       color="green")
print("ATB surprisals plotted.")

# calculate surprisal on atb training set
atb_testing_surprisals = []
for i, atb_tuple in enumerate(atb_testing):
    atb_testing_surprisals.append(
        surprisal.surprisal_effect_full_tuple(atb_tuple, "gpt2", True))

# plot data
plt.figure(2)
build_figure_from_data(surprisal_effects=atb_testing_surprisals,
                       plot_title="ATB Surprisals: GPT2 | Testing",
                       color="green")
print("ATB surprisals plotted.")

# show plots
plt.show()

# save corpuses (which should have surprisal!)
generate_corpora.corpus_to_json(atb_training, "atb_training.json")
generate_corpora.corpus_to_json(atb_testing, "atb_testing.json")
"""


atb_training = generate_corpora. corpus_from_json(
    where_to_load='atb_training.json', is_tuples=True)

training_surprisals = [
    surprisal.compute_surprisal_effect_from_surprisals
    (
        tuple.s_fg.critical_surprisal,
        tuple.s_xg.critical_surprisal,
        tuple.s_fx.critical_surprisal,
        tuple.s_xx.critical_surprisal
    ) for tuple in atb_training
]

build_figure_from_data(surprisal_effects=training_surprisals,
                       plot_title="ATB Training Surprisals Loaded from JSON", color="red")
plt.show()
