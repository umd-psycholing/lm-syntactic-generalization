import csv
import os
import json

from surprisal_utilities import gpt2_surprisal, grnn_surprisal

script_directory = os.path.dirname(os.path.abspath(__file__))
csv_input_directory = os.path.join(
    script_directory, '..', 'data', 'cfg-output', 'tuples')
output_directory = os.path.join(script_directory,
                                '..', 'data/surprisal_jsons')

config_name = [filename.removesuffix('.csv')
               for filename in os.listdir(csv_input_directory)
               if filename.endswith('csv')]


def surprisal_json_at(input_path, output_path, function: callable):
    data_dict = {}

    with open(input_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            sentence = row['Sentence']

            # don't do duplicate sentences!
            if sentence in data_dict.keys():
                continue

            type = row['Type'].removeprefix('<').removesuffix('>')
            critical = row.get('Critical String')

            surprisals = {token: surprisal
                          for token, surprisal in function(sentence)}
            data_dict[sentence] = {
                "type": type,
                "surprisals": surprisals,
                "critical": critical,
            }

            # print(f"Sentnce: {sentence} added!")

    with open(output_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    print(f"{output_path} done!")


# do it for gpt2
# [surprisal_json_at(
#     os.path.join(csv_input_directory, f'{config_path}.csv'),
#     os.path.join(output_directory, 'gpt2',
#                  f'{config_path.removesuffix("tuple_output")}gpt2_by_word_surprisal.json'),
#     gpt2_surprisal)
#  for config_path in config_name]

# print("DONE GPT2")

# do it for grnn (untrained)
[surprisal_json_at(
    os.path.join(csv_input_directory, f'{config_path}.csv'),
    os.path.join(output_directory, 'grnn_intrained',
                 f'{config_path.removesuffix("tuple_output")}grnn_by_word_surprisal.json'),
    grnn_surprisal)
 for config_path in config_name]

print("DONE GRNN")
