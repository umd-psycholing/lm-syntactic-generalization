import pandas as pd
from grammar_utilities import generate_sentences
import os
import json
from itertools import product
import csv


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


def build_tuple_csv_at(config_path, output_path=None):
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
output_directory = os.path.join(
    script_directory, '..', 'data/cfg-output/tuples')


config_name = [filename.removesuffix('.json')
               for filename in os.listdir(config_directory)
               if filename.endswith('json')]


[build_tuple_csv_at(
    os.path.join(config_directory, f'{config_path}.json'),
    os.path.join(output_directory, f'{config_path}_tuple_output.csv'))
 for config_path in config_name]


def group_sentences(csv_filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filename)

    # Initialize empty list to store grouped sentences
    grouped_sentences = []

    # Iterate through the DataFrame in groups of four rows
    for i in range(0, len(df), 4):
        grouped_sentence = {
            'S_FG': None,
            'S_XG': None,
            'S_FX': None,
            'S_XX': None
        }

        # Extract data for each sentence type
        for j, sentence_type in enumerate(['S_FG', 'S_XG', 'S_FX', 'S_XX']):
            sentence_data = df.iloc[i + j]
            grouped_sentence[sentence_type] = {
                'Sentence': sentence_data['Sentence'],
                'Critical String': sentence_data['Critical String']
            }

        # Append the grouped sentence to the list
        grouped_sentences.append(grouped_sentence)

    return grouped_sentences
