from grammar_utilities import generate_sentences
import os
import json
from itertools import product
import matplotlib.pyplot as plt
from minicons import scorer
import csv

"""
wh_gap = "I know what John looked for yesterday and will devour tomorrow"
that_gap = "I know that John looked for food yesterday and will devour tomorrow."
gap_critical = "tomorrow"

wh_nogap = "I know what John looked for yesterday and will devour it tomorrow"
that_nogap = "I know that John looked for food yesterday and will devour it tomorrow."
nogap_critical = "it"

model = scorer.IncrementalLMScorer("gpt2")


def delta_delta_from_tuple(model, wh_gap, wh_nogap, that_gap, that_nogap, gap_critical, nogap_critical):
    # delta -filler - delta +filler
    plus_filler = compute_delta(
        model, that_gap, gap_critical, that_nogap, nogap_critical)
    minus_filler = compute_delta(
        model, wh_gap, gap_critical, wh_nogap, nogap_critical)

    return plus_filler - minus_filler


def compute_delta(model, gap_sentence: str, gap_critical: str, nogap_sentence: str, nogap_critical: str):
    # return surprisal(gap_sentence, gap_critical) - surprisal(nogap_sentence, nogap_critical)
    surprisals = model.token_score(
        [gap_sentence, nogap_sentence], surprisal=True, base_two=True)
    gap_surprisals = surprisals[0]
    gap_crit_surprisal = [token_score[1] for token_score in gap_surprisals
                          if token_score[0] == gap_critical][0]
    nogap_surprisals = surprisals[1]
    nogap_crit_surprisal = [token_score[1] for token_score in nogap_surprisals
                            if token_score[0] == nogap_critical][0]

    return gap_crit_surprisal - nogap_crit_surprisal


print(delta_delta_from_tuple(
    model,
    wh_gap, wh_nogap,
    that_gap, that_nogap,
    gap_critical, nogap_critical
))
"""


# new issue is building tuples...
# when generating tuples, start by deciding on a permutation of all the reserved types. THEN, the constructions should be deterministic, and it will produce four sentences for you.
# do that for each permutation of reserved types and you should be in business!


def construct_quad(grammar, reserved_permutation, reserved_types, starts) -> int:
    results = []

    # build new grammar
    new_grammar = grammar
    for i, reserved_type in enumerate(reserved_types):
        new_grammar[reserved_type] = [reserved_permutation[i]]

    # build new tuple of sentences
    sentence_tuple = []
    for start in starts:
        type, sentence_list = generate_sentences(new_grammar, start)
        sentence = sentence_list[0]
        sentence_tuple.append((type, sentence))

    for type, sentence in sentence_tuple:
        results.append([type, sentence.text, sentence.grammatical,
                        sentence.text[sentence.region_start:sentence.region_end],
                        sentence.region_start, sentence.region_end,])

    return results


def build_revised_csv_at(config_path, output_path=None):
    cfg = {}
    starts = []
    reserved_types = []
    with open(config_path) as input:
        config_json = json.load(input)

        cfg = config_json.get('grammar')
        starts = config_json.get('starts')
        reserved_types = config_json.get('reserved_types')

    options = []
    for reserved_type in reserved_types:
        options.append(cfg.get(reserved_type))

    permutations = list(product(*options))

    results = [("Type", "Sentence", "Grammatical",
                "Critical String", "Region Start", "Region End")]

    for permutation in permutations:
        word_tuple = construct_quad(
            cfg, permutation, reserved_types, starts)

        results.extend(word_tuple)

    # build it
    try:
        with open(output_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(results)

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while building the CSV {str(e)}")

 # generate 4 sentences per each permutation
script_directory = os.path.dirname(os.path.abspath(__file__))
config_directory = os.path.join(script_directory, 'cfg_configs')
output_directory = os.path.join(script_directory, '..', 'data/cfg-output/')


config_name = [filename.removesuffix('.json')
               for filename in os.listdir(config_directory)
               if filename.endswith('json')]


[build_revised_csv_at(
    os.path.join(config_directory, f'{config_path}.json'),
    os.path.join(output_directory, f'{config_path}_tuple_output.csv'))
 for config_path in config_name]
