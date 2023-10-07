import numpy as np
import multiprocessing
import os
import pandas as pd
import matplotlib.pyplot as plt
from minicons import scorer


wh_gap = "I know what John looked for yesterday and will devour tomorrow"
that_gap = "I know that John looked for food yesterday and will devour tomorrow."
gap_critical = "tomorrow"

wh_nogap = "I know what John looked for yesterday and will devour it tomorrow"
that_nogap = "I know that John looked for food yesterday and will devour it tomorrow."
nogap_critical = "it"

model = scorer.IncrementalLMScorer("gpt2")


def delta_delta_from_group(tuple: dict[str, dict[str, str]], model=model):
    wh_gap = tuple['S_FG']['Sentence']
    that_gap = tuple['S_XG']['Sentence']
    gap_critical = tuple['S_FG']['Critical String']

    wh_nogap = tuple['S_FX']['Sentence']
    that_nogap = tuple['S_XX']['Sentence']
    nogap_critical = tuple['S_FX']['Critical String']

    return delta_delta_direct(model,
                              wh_gap, wh_nogap, that_gap, that_nogap,
                              gap_critical, nogap_critical)


def delta_delta_direct(model, wh_gap, wh_nogap, that_gap, that_nogap, gap_critical, nogap_critical):
    # delta -filler - delta +filler
    minus_filler = _compute_delta(
        model, that_gap, gap_critical, that_nogap, nogap_critical)
    plus_filler = _compute_delta(
        model, wh_gap, gap_critical, wh_nogap, nogap_critical)

    return minus_filler - plus_filler


def _compute_delta(model, gap_sentence: str, gap_critical: str, nogap_sentence: str, nogap_critical: str):
    # return surprisal(gap_sentence, gap_critical) - surprisal(nogap_sentence, nogap_critical)
    surprisals = model.token_score(
        [gap_sentence, nogap_sentence], surprisal=True, base_two=True)
    gap_surprisals = surprisals[0]
    gap_crit_surprisal = [token_score[1] for token_score in gap_surprisals
                          if token_score[0] == gap_critical][0]
    nogap_surprisals = surprisals[1]
    nogap_crit_surprisal = [token_score[1] for token_score in nogap_surprisals
                            if token_score[0] == nogap_critical][0]

    return gap_crit_surprisal - nogap_crit_surprisal


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


script_directory = os.path.dirname(os.path.abspath(__file__))
config_directory = os.path.join(script_directory, 'cfg_configs')
output_directory = os.path.join(script_directory, '..', 'data/cfg-output/')

tuple_files = [os.path.join(output_directory, filename)
               for filename in os.listdir(output_directory)
               if filename.endswith('tuple_output.csv') and 'A.2' in filename]


grouped_sentences = []

for file in tuple_files:
    grouped_sentences.extend(group_sentences(file))

print(len(grouped_sentences))

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

    results = pool.map(delta_delta_from_group, grouped_sentences)

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
