from grammar_utilities import generate_sentences
import os
import json
from itertools import product
import csv


def construct_quad(grammar, lexical_permutation, lexical_types, starts, test_reserved, test_disallowed) -> int:
    results = []

    #########################################################################
    # Awaiting response from Lan to email regarding classifying sentences.  #
    # The method described (as is implemented below) results in far too few #
    # sentences in the training set, and WAY too many non-classified.       #
    #########################################################################

    # get set
    # count number of test-reserved, test-disallowed in permutation
    num_reserved = num_disallowed = 0
    for lexical_item in lexical_permutation:
        if lexical_item in test_reserved:
            num_reserved += 1
        elif lexical_item in test_disallowed:
            num_disallowed += 1
    if num_disallowed == 0:  # all must be test_reserved
        set = "Test"
    elif num_reserved < 2:  # no co-occurences of test set items
        set = "Training"
    else:
        set = "No Set"

    # build new grammar
    new_grammar = grammar
    for i, reserved_type in enumerate(lexical_types):
        new_grammar[reserved_type] = [lexical_permutation[i]]

    # build new tuple of sentences
    sentence_tuple = []
    for start in starts:
        type, sentence_list = generate_sentences(new_grammar, start)
        sentence = sentence_list[0]
        sentence_tuple.append((type, sentence))

    for type, sentence in sentence_tuple:
        results.append([type, sentence.text, sentence.grammatical, set,
                        sentence.text[sentence.region_start:sentence.region_end],
                        sentence.region_start, sentence.region_end,])

    return results


def build_tuple_csv_at(config_path, output_path=None):
    grammar = {}
    starts = []
    lexical_types = []
    with open(config_path) as input:
        config_json = json.load(input)

        grammar = config_json.get('grammar')
        starts = config_json.get('starts')
        lexical_types = config_json.get('lexical_types')

    # reserve testing set items
    test_reserved_lexical_items = []
    test_disallowed_lexical_items = []
    for lexical_type in lexical_types:
        non_terminals = grammar[lexical_type]
        test_reserved_lexical_items.extend(
            non_terminals[:round(len(non_terminals) * .65)])
        test_disallowed_lexical_items.extend(
            non_terminals[round(len(non_terminals) * .65):])

    options = []
    for reserved_type in lexical_types:
        options.append(grammar.get(reserved_type))

    permutations = list(product(*options))

    results = [("Type", "Sentence", "Grammatical", "Set",
                "Critical String", "Region Start", "Region End")]

    for permutation in permutations:
        word_tuple = construct_quad(
            grammar, permutation, lexical_types, starts,
            test_reserved_lexical_items, test_disallowed_lexical_items)

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
output_directory = os.path.join(
    script_directory, '..', 'data', 'cfg-output', 'tuples')


config_name = [filename.removesuffix('.json')
               for filename in os.listdir(config_directory)
               if filename.endswith('json')]


[build_tuple_csv_at(
    os.path.join(config_directory, f'{config_path}.json'),
    os.path.join(output_directory, f'{config_path}_tuple_output.csv'))
 for config_path in config_name]
