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
    corpus_deltas = [surprisal.compute_surprisal_effect_from_surprisals(s_tuple.s_fg.critical_surprisal,
                                                                        s_tuple.s_xg.critical_surprisal,
                                                                        s_tuple.s_fx.critical_surprisal,
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


# generate_atb_pg_train_test()
# print("Generated ATB, PG training & testing sets with default seed.")

# add surprisals to the corpuses
# extend_surprisal_corpus("ex/pg_atb_examples/s_atb_test.json",
#                         "ex/pg_atb_examples/s_atb_test_grnn.json", "grnn")
#
# extend_surprisal_corpus("ex/pg_atb_examples/s_atb_train.json",
#                         "ex/pg_atb_examples/s_atb_train_grnn.json", "grnn")
#
# extend_surprisal_corpus("ex/pg_atb_examples/s_pg_test.json",
#                         "ex/pg_atb_examples/s_pg_test_grnn.json", "grnn")
#
# extend_surprisal_corpus("ex/pg_atb_examples/s_pg_train.json",
#                         "ex/pg_atb_examples/s_pg_train_grnn.json", "grnn")


# load from corpuses (with surprisal) and display data
display_surprisal_corpus_data(
    "grammar_outputs/ex/pg_atb_examples/s_atb_test_grnn.json", "ATB Testing GRNN")

display_surprisal_corpus_data(
    "grammar_outputs/ex/pg_atb_examples/s_atb_train_grnn.json", "ATB Training GRNN")

display_surprisal_corpus_data(
    "grammar_outputs/ex/pg_atb_examples/s_pg_test_grnn.json", "PG Testing GRNN")

display_surprisal_corpus_data(
    "grammar_outputs/ex/pg_atb_examples/s_pg_train_grnn.json", "PG Training GRNN")


plt.show()
