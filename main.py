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


# generate all atb sentence tuples
atb_tuples = []
for i, text_grammar in enumerate(grammars.ATB_GRAMMARS):
    grammar = CFG.fromstring(text_grammar)
    atb_tuples.extend(
        generate_corpora.generate_all_sentence_tuples_from_grammar(grammar))
    print(f"All 2x2s generated for ATB Grammar {i + 1}.")

# calculate surprisal on all atb sentence tuples
atb_surprisals = []
for i, atb_tuple in enumerate(atb_tuples):
    atb_surprisals.append(
        surprisal.surprisal_effect_from_tuple(atb_tuple, "gpt2"))
    if i % 100 == 99:
        print(f"Surprisals generated for ATB Grammar {i + 1}.")

# plot data
plt.figure(1)
build_figure_from_data(surprisal_effects=atb_surprisals,
                       plot_title="ATB Surprisals: GPT2 | All 2x2s",
                       color="green")
print("ATB surprisals plotted.")

# generate all pg sentence tuples
pg_tuples = []
for i, text_grammar in enumerate(grammars.PG_GRAMMARS):
    grammar = CFG.fromstring(text_grammar)
    pg_tuples.extend(
        generate_corpora.generate_all_sentence_tuples_from_grammar(grammar))
    print(f"All 2x2s generated for PG Grammar {i + 1}.")

# calculate surprisal on all pg sentence tuples
pg_surprisals = []
for i, pg_tuple in enumerate(pg_tuples):
    pg_surprisals.append(
        surprisal.surprisal_effect_from_tuple(pg_tuple, "gpt2"))
    if i % 100 == 99:
        print(f"Surprisals generated for PG Grammar {i + 1}.")

# plot data
plt.figure(2)
build_figure_from_data(surprisal_effects=pg_surprisals,
                       plot_title="PG Surprisals: GPT2 | All 2x2s",
                       color="green")
print("PG surprisals plotted.")

# show plots
plt.show()
