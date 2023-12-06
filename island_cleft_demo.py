import generate_corpora as gc
import surprisal
import grammars

import numpy as np
from nltk.parse import generate
import matplotlib.pyplot as plt


# load
island_sentence_tuples = gc.corpus_from_json(
    "grammar_outputs/island_cleft/tuples.json", True)

# add surprisals
[surprisal.surprisal_effect_full_tuple(sentence_tuple, "gpt2", True)
 for sentence_tuple in island_sentence_tuples]

# save
gc.corpus_to_json(island_sentence_tuples,
                  "grammar_outputs/island_cleft/tuples_gpt2.json")


# load
tuple_data = gc.corpus_from_json(
    "grammar_outputs/island_cleft/tuples_gpt2.json", is_tuples=True)

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
