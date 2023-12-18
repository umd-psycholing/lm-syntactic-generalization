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


def wilcox_replication(file_path: str, nogap_critical_keys: tuple[str], gap_critical_keys: tuple[str], gap_location: str):
    # subject
    wh_tuples = []
    with open(f'wilcox_csv/{file_path}') as file:
        csv_reader = list(csv.DictReader(file))

        items = [line.get('item') for line in csv_reader]
        # remove duplicates w/out losing ordering by converting to a set
        items = [i for n, i in enumerate(items) if i not in items[:n]]

        for i in items:  # works for 4-way comparisons, ex: +/-filler, +/-gap
            sentence_datas = {}
            entire_item = [d for d in csv_reader if d.get('item') == i]
            for sentence in entire_item:
                if sentence['condition'] == 'what_gap':
                    sentence_datas["s_fg"] = gc._grammar_output_to_sentence(
                        _critical_keys(sentence, gap_critical_keys[0], gap_critical_keys[1])[2:])
                elif sentence['condition'] == 'that_gap':
                    sentence_datas["s_xg"] = gc._grammar_output_to_sentence(
                        _critical_keys(sentence, gap_critical_keys[0], gap_critical_keys[1])[2:])
                elif sentence['condition'] == 'what_nogap':
                    sentence_datas["s_fx"] = gc._grammar_output_to_sentence(
                        _critical_keys(sentence, nogap_critical_keys[0], nogap_critical_keys[1])[2:])
                elif sentence['condition'] == 'that_nogap':
                    sentence_datas["s_xx"] = gc._grammar_output_to_sentence(
                        _critical_keys(sentence, nogap_critical_keys[0], nogap_critical_keys[1])[2:])
            wh_tuples.append(
                gc.TupleSentenceData(s_ab=sentence_datas['s_fg'],
                                     s_xb=sentence_datas['s_xg'],
                                     s_ax=sentence_datas['s_fx'],
                                     s_xx=sentence_datas['s_xx']
                                     )
            )

    # calculate all surprisals
    for wh_tuple in wh_tuples:
        surprisal.surprisal_effect_full_tuple(wh_tuple, model, True)

    gc.corpus_to_json(wh_tuples,
                      f"wilcox_csv/wilcox_{gap_location}_wh_{model}.json")

    # load
    gap_tuple_data = gc.corpus_from_json(
        f"wilcox_csv/wilcox_{gap_location}_wh_{model}.json", is_tuples=True)

    # extract
    gap_wh_effects = []
    nogap_wh_effects = []
    for sentence_tuple in gap_tuple_data:
        # effect of wh when there *is* a gap (+gap wh-effect)
        gap_wh_effects.append(  # (wh, ___) - (that, ___)
            sentence_tuple.s_ab.critical_surprisal - sentence_tuple.s_xb.critical_surprisal)
        # effect of wh when there is *not* a gap (-gap wh-effect)
        nogap_wh_effects.append(  # (wh, no gap) - (that, no gap)
            sentence_tuple.s_ax.critical_surprisal - sentence_tuple.s_xx.critical_surprisal)

    # average
    avg_gap_wh_effect = np.median(gap_wh_effects)
    avg_nogap_wh_effect = np.median(nogap_wh_effects)

    # plot
    fig, ax = plt.subplots()
    ax.bar(
        [
            f"{gap_location} +gap wh-effect",
            f"{gap_location} -gap wh-effect"
        ],
        [
            avg_gap_wh_effect,
            avg_nogap_wh_effect
        ]
    )


# subject
wilcox_replication('basic_subject.csv', ('np1', 'np1'),
                   ('verb', 'verb'), "subject")

# object
wilcox_replication('basic_object.csv', ('np2', 'np2'),
                   ('prep', 'np3'), "object")

# prepositional phrase
wilcox_replication('basic_pp.csv', ('np3', 'np3'),
                   ('end', 'end'), "pp")

plt.show()
