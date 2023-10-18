import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
csv_tuple_directory = os.path.join(
    script_directory, '..', 'data', 'cfg-output', 'tuples')
gpt_surprisal_json_directory = os.path.join(script_directory,
                                            '..', 'data/surprisal_jsons', 'gpt2')


def group_sentences(csv_filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filename)

    grouped_sentences = []

    # Iterate through the DataFrame in groups of four
    for i in range(0, len(df), 4):
        group = df.iloc[i:i+4]

        group_dict = {}
        for _, row in group.iterrows():
            group_dict[row["Type"]] = row["Sentence"]

        grouped_sentences.append(group_dict)

    return grouped_sentences


def compute_delta_deltas(surprisal_json_input, tuple_csv_input):
    surprisal_dict = {}
    with open(surprisal_json_input, 'r') as json_file:
        surprisal_dict = json.load(json_file)

    grouped_sentences = group_sentences(tuple_csv_input)

    delta_deltas = []

    for group in grouped_sentences:
        sentence_FG = group.get('<S_FG>')
        sentence_XG = group.get('<S_XG>')
        sentence_FX = group.get('<S_FX>')
        sentence_XX = group.get('<S_XX>')

        surprisal_FG = surprisal_dict[sentence_FG]["surprisals"][surprisal_dict[sentence_FG]["critical"]]
        surprisal_XG = surprisal_dict[sentence_XG]["surprisals"][surprisal_dict[sentence_XG]["critical"]]
        surprisal_FX = surprisal_dict[sentence_FX]["surprisals"][surprisal_dict[sentence_FX]["critical"]]
        surprisal_XX = surprisal_dict[sentence_XX]["surprisals"][surprisal_dict[sentence_XX]["critical"]]

        delta_filler = surprisal_FG - surprisal_FX
        delta_no_filler = surprisal_XG - surprisal_XX

        delta_deltas.append(delta_no_filler - delta_filler)

    return delta_deltas


# We assume that the surprisals have already been calculated, and that all tuples are generated

# Calcualate delta-filler minus delta+filler for both PG and ATB (for gpt)
pg_surprisals = []
for tuple_csv, surprisal_json in zip(os.listdir(csv_tuple_directory)[1:4],
                                     os.listdir(gpt_surprisal_json_directory)[1:4]):
    tuple_path = os.path.join(csv_tuple_directory, tuple_csv)
    surprisal_path = os.path.join(gpt_surprisal_json_directory, surprisal_json)

    pg_surprisals.extend(compute_delta_deltas(
        tuple_csv_input=tuple_path,
        surprisal_json_input=surprisal_path
    ))

atb_surprisals = []
for tuple_csv, surprisal_json in zip(os.listdir(csv_tuple_directory)[4:],
                                     os.listdir(gpt_surprisal_json_directory)[4:]):
    tuple_path = os.path.join(csv_tuple_directory, tuple_csv)
    surprisal_path = os.path.join(gpt_surprisal_json_directory, surprisal_json)

    atb_surprisals.extend(compute_delta_deltas(
        tuple_csv_input=tuple_path,
        surprisal_json_input=surprisal_path
    ))


# Plotting data
num_pg = np.arange(len(pg_surprisals))
num_pg_greater_than_zero = sum(1 for num in pg_surprisals if num > 0)
plt.figure(1)
plt.scatter(num_pg, pg_surprisals, s=2, alpha=0.5, color='green')
plt.xlabel(
    f'{round(num_pg_greater_than_zero / len(pg_surprisals), 2)} > 0 ({num_pg_greater_than_zero} / {len(pg_surprisals)})')
plt.ylabel('Δ-filler - Δ+filler')
plt.title('PG surprisals')
plt.axhline(linestyle='-', label='y=0', color='black')

num_atb = np.arange(len(atb_surprisals))
num_atb_greater_than_zero = sum(1 for num in atb_surprisals if num > 0)
plt.figure(2)
plt.scatter(num_atb, atb_surprisals, s=2, alpha=0.5, color='green')
plt.xlabel(
    f'{round(num_atb_greater_than_zero / len(atb_surprisals), 2)} > 0 ({num_atb_greater_than_zero} / {len(atb_surprisals)})')
plt.ylabel('Δ-filler - Δ+filler')
plt.title('ATB surprisals')
plt.axhline(linestyle='-', label='y=0', color='black')

plt.show()

#######################################################################################
# Still not sure how to limit the number of tuples. I end up w/ upwards of 32k tuples #
# (generated using all possible fully-qualified <S_XX> sentences from provided cfg's) #
# while Lan et al. only have 6,144 PG data-points & 5552 GPT2 data points.            #
#######################################################################################
