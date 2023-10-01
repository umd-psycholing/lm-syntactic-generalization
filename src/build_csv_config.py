from grammar_utilities import build_csv
import os
import json

script_directory = os.path.dirname(os.path.abspath(__file__))
config_directory = os.path.join(script_directory, 'cfg_configs')
output_directory = os.path.join(script_directory, '..', 'data', 'cfg-output')

for filename in os.listdir(config_directory):
    if filename.endswith('.json'):
        filename = filename.removesuffix('.json')
        config_path = os.path.join(config_directory, f'{filename}.json')

        with open(config_path) as input:
            config = json.load(input)

        cfg = config['grammar']
        starts = config['starts']

        # Construct the output file path relative to the script
        output_file_path = os.path.join(
            output_directory, f'{filename}_output.csv')

        build_csv(cfg, starts, output_file_path)

print("complete!")
