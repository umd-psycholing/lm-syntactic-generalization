from typing import Iterable
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from nltk.grammar import CFG
from nltk.parse import generate

import generate_corpora as gc
import surprisal
import grammars


# handles matplotlib
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


# load surprisal-filled corpus and plot data
def display_surprisal_corpus_data(corpus_filepath: str, plot_title: str = "Corpus Data"):
    corpus_deltas = [surprisal.compute_surprisal_effect_from_surprisals(s_tuple.s_ab.critical_surprisal,
                                                                        s_tuple.a_xb.critical_surprisal,
                                                                        s_tuple.s_ax.critical_surprisal,
                                                                        s_tuple.s_xx.critical_surprisal)
                     for s_tuple in gc.corpus_from_json(corpus_filepath, is_tuples=True)]  # load tuples
    plt.figure()
    build_figure_from_data(corpus_deltas, plot_title)


# demonstration purposes, generate atb, pg training and testing data & save them
# saved to grammar_outputs/train_test_sets
def generate_atb_pg_train_test(where_to_save: str = "grammar_outputs/ex/pg_atb_examples"):
    atb_train = []
    atb_test = []
    for i, grammar in enumerate(grammars.ATB_GRAMMARS):
        grammar = CFG.fromstring(grammar)

        train, test = gc.generate_train_test_tuples_from_grammar(
            grammar=grammar, split_ratio=0.65)

        atb_train.extend(train)
        atb_test.extend(test)

    gc.corpus_to_json(
        atb_train, f"{where_to_save}/s_atb_train.json")
    gc.corpus_to_json(
        atb_test, f"{where_to_save}/s_atb_test.json")

    pg_train = []
    pg_test = []
    for i, grammar in enumerate(grammars.PG_GRAMMARS):
        grammar = CFG.fromstring(grammar)

        train, test = gc.generate_train_test_tuples_from_grammar(
            grammar=grammar, split_ratio=0.65)

        pg_train.extend(train)
        pg_test.extend(test)

    gc.corpus_to_json(
        pg_train, f"{where_to_save}/s_pg_train.json")
    gc.corpus_to_json(
        pg_test, f"{where_to_save}/s_pg_test.json")


# add surprisals to the corpus
def extend_surprisal_corpus(where_to_load: str, where_to_save: str, model: str):
    corpus = gc.corpus_from_json(where_to_load)
    surprisals = []
    for i, tup in enumerate(corpus):
        surprisals.append(
            surprisal.surprisal_effect_full_tuple(tup, model, True))
    gc.corpus_to_json(where_to_save)


# load
island_sentence_tuples = gc.corpus_from_json(
    "grammar_outputs/island/island_tuples.json", True)

# add surprisals
[surprisal.surprisal_effect_full_tuple(sentence_tuple, "grnn", True)
 for sentence_tuple in island_sentence_tuples]

# save
gc.corpus_to_json(island_sentence_tuples,
                  "grammar_outputs/island/island_tuples_grnn.json")


# load
tuple_data = gc.corpus_from_json(
    "grammar_outputs/island/island_tuples_grnn.json", is_tuples=True)

# extract
island_gap_effects = []
islandless_gap_effects = []
for sentence_tuple in tuple_data:
    island_gap_effects.append(  # (island, no gap) - (island, gap)
        sentence_tuple.s_xb.critical_surprisal - sentence_tuple.s_xx.critical_surprisal
    )
    islandless_gap_effects.append(  # (no island, no gap) - (no island, gap)
        sentence_tuple.s_ab.critical_surprisal - sentence_tuple.s_ax.critical_surprisal)

# average
avg_island_gap_effect = np.mean(island_gap_effects)
avg_islandless_gap_effect = np.mean(islandless_gap_effects)

# plot
fig, ax = plt.subplots()
ax.bar(
    [
        "Effect of Extractible Gap (-Island)",
        "Effect of Stranded Gap (+Island)"
    ],
    [
        avg_islandless_gap_effect,
        avg_island_gap_effect
    ]
)

plt.show()
