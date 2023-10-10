import csv
import os
import json

from surprisal_utilities import gpt2_surprisal, grnn_surprisal

script_directory = os.path.dirname(os.path.abspath(__file__))
csv_input_directory = os.path.join(
    script_directory, '..', 'data/cfg-output/')
output_directory = os.path.join(script_directory,
                                '..', 'data/surprisal_jsons')

config_name = [filename.removesuffix('.csv')
               for filename in os.listdir(csv_input_directory)
               if filename.endswith('csv')]


def surprisal_json_at(input_path, output_path):
    count = 0
    data_dict = {}

    with open(input_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            sentence = row['Sentence']
            type = row['Type'].removeprefix('<').removesuffix('>')
            critical = row['Critical String']

            surprisals = {token: surprisal
                          for token, surprisal in gpt2_surprisal(sentence)}
            data_dict[sentence] = {
                "type": type,
                "surprisals": surprisals,
                "critical": critical,
            }

    with open(output_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    print(f"{output_path} done!")


[surprisal_json_at(
    os.path.join(csv_input_directory, f'{config_path}.csv'),
    os.path.join(output_directory, f'{config_path.removesuffix("output")}gpt2_by_word_surprisal.json'))
 for config_path in config_name]
