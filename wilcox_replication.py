import argparse
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import generate_corpora as gc
from surprisal import surprisal_effect_full_tuple

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices = ['grnn', 'gpt2'])
    parser.add_argument("--data", help = 'path to data')
    parser.add_argument("--gap_region", nargs=2, help = 'region to compute surprisal for sentences with gaps')
    parser.add_argument("--nogap_region", nargs=2, help = 'region to compute surprisal for sentences without gaps')
    parser.add_argument("--sentence_type", choices = ['basic_subj', 'basic_obj', 'basic_pp', 'island_cnp'])
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    gap_critical_keys, nogap_critical_keys = tuple(args.gap_region), tuple(args.nogap_region)
    items = df['item'].unique()
    tuples = setup_tuple_dict(df, args.sentence_type)
    print(f"processing {args.sentence_type}")
    group_sentences(df, tuples, items, gap_critical_keys, nogap_critical_keys)
    save_surprisal(tuples, args.sentence_type, args.model)

def group_sentences(df, tuples, items, gap_critical_keys, nogap_critical_keys):
    for item_id in items:
        sentences = df[df['item'] == item_id]
        basic = len(sentences) == 4
        for condition_type in tuples.keys():
            if basic:
                condition_type = "" # yes this is hacky since we're just doing gap vs nogap instead of control vs island. can probably change around?
            fourway_tuple = sentences[sentences['condition'].str.endswith(condition_type)]
            assert len(fourway_tuple) == 4
            fourway_tuple = fourway_tuple.to_dict(orient = 'records')
            extracted_item = {}
            for sentence in fourway_tuple:
                extraction_key = sentence['condition'] # wh_gap, that_gap, wh_nogap, that_nogap
                if not basic:
                    extraction_key = sentence['condition'].replace(f"_{condition_type}", "")
                extracted_item[extraction_key] = gc._grammar_output_to_sentence(
                            (_critical_keys(sentence, gap_critical_keys[0], gap_critical_keys[1]) if 'nogap' not in sentence['condition'] 
                            else _critical_keys(sentence, nogap_critical_keys[0], nogap_critical_keys[1]))[1:])
            if basic:
                condition_type = "wh"
            tuples[condition_type].append(
                gc.TupleSentenceData(
                    s_ab=extracted_item['what_gap'], # filler, gap
                    s_xb=extracted_item['that_gap'], # no filler, gap
                    s_ax=extracted_item['what_nogap'], # filler, no gap
                    s_xx=extracted_item['that_nogap'], # no filler, no gap
                )
            )

def _critical_keys(d, from_key, to_key):
    keys_list = list(d.keys())
    values_list = list(d.values())

    # Find the indices of the provided keys
    index1 = keys_list.index(from_key)
    index2 = keys_list.index(to_key)

    values_list.insert(index2 + 1, "_")
    values_list.insert(index1, "_")
    # remove empties
    values_list = [value for value in values_list if type(value) == str and len(value) > 0]

    return values_list

def setup_tuple_dict(df: pd.DataFrame, sentence_type: str):
    # each type is . Note: for other basic conditions we can change the key, Wilcox et al just uses wh-licensing
    tuples = {}
    if "basic" in sentence_type:
        tuples['wh'] = []
    else:
        condition_types = np.unique([condition.split("_")[-1] for condition in df['condition']])
        for condition_type in condition_types:
            tuples[condition_type] = []
    return tuples

def save_surprisal(tuples: Dict[str, List[gc.TupleSentenceData]], sentence_type: str, model: str):
    # compute surprisals and save
    for sentence_set in tuples.keys():
        print(f"computing surprisal for {sentence_type} {sentence_set}")
        for sentence_tuple in tqdm(tuples[sentence_set]):
            surprisal_effect_full_tuple(sentence_tuple, model, True)
        gc.corpus_to_json(tuples[sentence_set], f"grammar_outputs/wilcox_replication/cleft_rnn/{sentence_type}_{sentence_set}_{model}.json")

if __name__ == "__main__":
    main()
