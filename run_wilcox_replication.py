import subprocess
import torch

MODEL = "grnn"
if torch.__version__ >= "2.0":
    MODEL = "gpt2"
# we have to switch the Python env to use GPT2/GRNN, otherwise we could run model-wise evals.

run_configs = {
    'basic_subj': {'gap_region': ('verb', 'verb'),
                   'nogap_region': ('np1', 'np1'),
                   'data_location': 'data/wilcox_csv/basic_subject.csv'},
    'basic_obj': {'gap_region': ('prep', 'prep'),
                  'nogap_region': ('np2', 'np2'),
                  'data_location': 'data/wilcox_csv/basic_object.csv'},
    'basic_pp': {'gap_region': ('end', 'end'),
                 'nogap_region': ('np3', 'np3'),
                 'data_location': 'data/wilcox_csv/basic_pp.csv'},
    'island_cnp': {'gap_region': ('continuation', 'continuation'),
                   'nogap_region': ('rc_obj', 'rc_obj'),
                   'data_location': 'data/wilcox_csv/islands_cnp.csv'}
}

for sentence_type in run_configs.keys():
    config = run_configs[sentence_type]
    subprocess.Popen(['python', 'wilcox_replication.py', '--model', MODEL, '--data', config['data_location'],
                      '--gap_region', config['gap_region'][0], config['gap_region'][1],
                      '--nogap_region', config['nogap_region'][0], config['nogap_region'][1],
                      '--sentence_type', sentence_type
                      ])
