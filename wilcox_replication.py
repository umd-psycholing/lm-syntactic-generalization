import csv
import generate_corpora as gc
import surprisal

import numpy as np
import matplotlib.pyplot as plt

model = "gpt2"


def _critical_keys(d, from_key, to_key):
    keys_list = list(d.keys())
    values_list = list(d.values())

    # Find the indices of the provided keys
    index1 = keys_list.index(from_key)
    index2 = keys_list.index(to_key)

    values_list.insert(index2 + 1, "_")
    values_list.insert(index1, "_")

    # remove empties
    values_list = [value for value in values_list if len(value) > 0]

    return values_list

# subject: critical region is either np1 or verb
# object: critical region is either np2 or prep,np3
# pp: critical region is either np3 or end


# subject
subject_wh_tuples = []
with open('wilcox_csv/basic_subject.csv') as file:
    csv_reader = list(csv.DictReader(file))

    for i in range(1, 50):
        sentence_datas = {}
        entire_item = [d for d in csv_reader if d.get('item') == str(i)]
        for sentence in entire_item:
            if sentence['condition'] == 'what_gap':
                sentence_datas["s_fg"] = gc._grammar_output_to_sentence(
                    _critical_keys(sentence, 'verb', 'verb')[2:])
            elif sentence['condition'] == 'that_gap':
                sentence_datas["s_xg"] = gc._grammar_output_to_sentence(
                    _critical_keys(sentence, 'verb', 'verb')[2:])
            elif sentence['condition'] == 'what_nogap':
                sentence_datas["s_fx"] = gc._grammar_output_to_sentence(
                    _critical_keys(sentence, 'np1', 'np1')[2:])
            elif sentence['condition'] == 'that_nogap':
                sentence_datas["s_xx"] = gc._grammar_output_to_sentence(
                    _critical_keys(sentence, 'np1', 'np1')[2:])
        subject_wh_tuples.append(
            gc.TupleSentenceData(s_ab=sentence_datas['s_fg'],
                                 s_xb=sentence_datas['s_xg'],
                                 s_ax=sentence_datas['s_fx'],
                                 s_xx=sentence_datas['s_xx']
                                 )
        )
# calculate all surprisals
for sw_tuple in subject_wh_tuples:
    surprisal.surprisal_effect_full_tuple(sw_tuple, model, True)

gc.corpus_to_json(subject_wh_tuples,
                  f"wilcox_csv/wilcox_subject_wh_{model}.json")


# load
sw_tuple_data = gc.corpus_from_json(
    f"wilcox_csv/wilcox_subject_wh_{model}.json", is_tuples=True)

# extract
gap_wh_effects = []
nogap_wh_effects = []
for sentence_tuple in sw_tuple_data:
    gap_wh_effects.append(  # (wh, gap) - (no wh, gap)
        sentence_tuple.s_ab.critical_surprisal - sentence_tuple.s_xb.critical_surprisal)
    nogap_wh_effects.append(  # (wh, no gap) - (no wh, no gap)
        sentence_tuple.s_ax.critical_surprisal - sentence_tuple.s_xx.critical_surprisal)

# average
avg_gap_wh_effect = np.mean(gap_wh_effects)
avg_nogap_wh_effect = np.mean(nogap_wh_effects)

# plot
fig, ax = plt.subplots()
ax.bar(
    [
        "Subject +gap wh-effect",
        "Subject -gap wh-effect"
    ],
    [
        avg_gap_wh_effect,
        avg_nogap_wh_effect
    ]
)
plt.show()
