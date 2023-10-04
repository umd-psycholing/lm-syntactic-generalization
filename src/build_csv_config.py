from grammar_utilities import build_csv
import os
import json


def build_csv_at(config_path, output_path=None):
    with open(config_path) as input:
        config_json = json.load(input)

        cfg = config_json['grammar']
        starts = config_json['starts']

        # default output directory at ../data/cfg-output/{config_name}_output.csv
        if not output_path:
            print("ok")
            output_path = os.path.join(output_directory,
                                       f'{config_path.removesuffix(".json")}_output.csv')

        build_csv(cfg, starts, output_path)


script_directory = os.path.dirname(os.path.abspath(__file__))

# default input directory at ./src/cfg_configs/{config_name}.json
config_directory = os.path.join(script_directory, 'cfg_configs')

# default output directory at ../data/cfg-output/{config_name}_output.csv
output_directory = os.path.join(script_directory, '..', 'data', 'cfg-output')

# list of config files
config_name = [filename.removesuffix('.json')
               for filename in os.listdir(config_directory)
               if filename.endswith('json')]


[build_csv_at(
    os.path.join(config_directory, f'{config_path}.json'),
    os.path.join(output_directory, f'{config_path}_output.csv'))
 for config_path in config_name]


print("complete!")
