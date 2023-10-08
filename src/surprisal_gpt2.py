from surprisal_utilities import delta_delta_from_group
from build_sentence_tuples import group_sentences

import numpy as np
import multiprocessing
import os
import matplotlib.pyplot as plt
from minicons import scorer


script_directory = os.path.dirname(os.path.abspath(__file__))
config_directory = os.path.join(script_directory, 'cfg_configs')
output_directory = os.path.join(script_directory, '..', 'data/cfg-output/')

tuple_files = [os.path.join(output_directory, filename)
               for filename in os.listdir(output_directory)
               if filename.endswith('tuple_output.csv') and 'A.2' in filename]


grouped_sentences = []

for file in tuple_files:
    grouped_sentences.extend(group_sentences(file))


def delta_delta_gpt2(grouped_sentence):
    return delta_delta_from_group(grouped_sentence, gpt2_model)


def delta_delta_grnn(grouped_sentence):
    return delta_delta_from_group(grouped_sentence, grnn_model)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

    results = pool.map(delta_delta_gpt2, grouped_sentences)

    pool.close()
    pool.join()

    total_delta_delta = sum(results)
    print(
        f"Average: {total_delta_delta / len(results)}\nMinimum: {min(results)}\nMaximum: {max(results)}")

    y_values = results

    x_values = np.arange(len(y_values))

    plt.scatter(x_values, y_values, s=3)
    plt.xlabel('X-values (Hidden)')
    plt.ylabel('delta-filler - delta+filler')
    plt.title('TITLE')
    plt.show()
