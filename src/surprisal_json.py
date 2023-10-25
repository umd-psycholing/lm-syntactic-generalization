import csv
import os
import json

import surprisal

script_directory = os.path.dirname(os.path.abspath(__file__))
csv_tuple_directory = os.path.join(
    script_directory, '..', 'data', 'cfg-output', 'tuples')
surprisal_json_directory = os.path.join(script_directory,
                                        '..', 'data/surprisal_jsons')

config_name = [filename.removesuffix('.csv')
               for filename in os.listdir(csv_tuple_directory)
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

            # find a way to handle second-instance of a sentence
            #   add an underline after for duplicates ex: "the" -> "the_" -> "the__" -> ...
            #   add number before/after for duplicates ex: "the" -> {"2_the"/"the_2"} -> {"3_the"/"the_3"} -> ...
            surprisals = {token: surprisal
                          for token, surprisal in function(sentence)}
            
            data_dict[sentence] = {
                "type": type,
                "surprisals": surprisals,
                "critical": critical,
            }

    with open(output_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    print(f"{output_path} done!")


# gpt2 ** (newer torch version) **
[surprisal_json_at(
    input_path=os.path.join(csv_tuple_directory, f'{config_path}.csv'),
    output_path=os.path.join(surprisal_json_directory, 'gpt2',
                 f'{config_path.removesuffix("tuple_output")}gpt2_by_word_surprisal.json'),
    function=surprisal.gpt2_surprisal)
 for config_path in config_name]
print("DONE GPT2")

# grnn (untrained) ** (older torch version) **
# [surprisal_json_at(
#     input_path=os.path.join(csv_tuple_directory, f'{config_path}.csv'),
#     output_path=os.path.join(surprisal_json_directory, 'grnn_untrained',
#                  f'{config_path.removesuffix("tuple_output")}grnn_by_word_surprisal.json'),
#     function=surprisal.grnn_surprisal)
#  for config_path in config_name]
# print("DONE GRNN")
