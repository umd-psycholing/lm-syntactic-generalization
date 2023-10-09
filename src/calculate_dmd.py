from surprisal_utilities import compute_delta_gap
from surprisal_utilities import gpt2_surprisal, grnn_surprisal
from build_sentence_tuples import group_sentences

import numpy as np
import multiprocessing
import os
import matplotlib.pyplot as plt
from minicons import scorer


# wrapper for delta minus delta with gpt2
def group_dmd_gpt2(tuple: dict[str, dict[str, str]]):
    return group_delta_minus_delta(tuple, gpt2_surprisal)


# wrapper for delta minus delta with grnn
def group_dmd_grnn(tuple: dict[str, dict[str, str]]):
    return group_delta_minus_delta(tuple, grnn_surprisal)


# unpack tuple sentences, compute surprisal using model_function
def group_delta_minus_delta(tuple: dict[str, dict[str, str]], model_func: function):
    wh_gap = tuple['S_FG']['Sentence']
    that_gap = tuple['S_XG']['Sentence']
    gap_critical = tuple['S_FG']['Critical String']

    wh_nogap = tuple['S_FX']['Sentence']
    that_nogap = tuple['S_XX']['Sentence']
    nogap_critical = tuple['S_FX']['Critical String']

    minus_filler = compute_delta_gap(
        model_func, that_gap, gap_critical, that_nogap, nogap_critical)
    plus_filler = compute_delta_gap(
        model_func, wh_gap, gap_critical, wh_nogap, nogap_critical)

    return minus_filler - plus_filler


# build tuples of sentences
script_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(
    script_directory, '..', 'data/cfg-output/tuples')

tuple_files = [os.path.join(output_directory, filename)
               for filename in os.listdir(output_directory)
               if filename.endswith('tuple_output.csv') and 'A.2' in filename]
grouped_sentences = []
for file in tuple_files:
    grouped_sentences.extend(group_sentences(file))


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

    results = pool.map(group_dmd_gpt2, grouped_sentences)

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
