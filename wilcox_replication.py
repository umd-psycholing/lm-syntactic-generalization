import csv
import generate_corpora as gc
import surprisal

import numpy as np
import matplotlib.pyplot as plt

# 'grnn' or 'gpt2'
model = "grnn"


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

def wilcox_cnp_licensing(file_path: str,
                         nogap_critical_keys, gap_critical_keys,
                         average_type: str = 'mean'):

    obj_tuples = []
    that_tuples = []
    wh_tuples = []

    with open(f'wilcox_csv/{file_path}') as file:
        csv_reader = list(csv.DictReader(file))

        items = max([int(line.get('item')) for line in csv_reader])

        for i in range(1, items): # for each set of 12 sentences
            all_sentences = [entry for entry in csv_reader if entry.get('item') == str(i)]
            
            extracted_item = {}

            for sentence in all_sentences:
                s_condition = sentence['condition']
                
                # gap keys if 'gap', otherwise nogap keys
                extracted_item[s_condition] = gc._grammar_output_to_sentence(
                    (_critical_keys(sentence, gap_critical_keys[0], gap_critical_keys[1]) if 'nogap' not in s_condition 
                    else _critical_keys(sentence, nogap_critical_keys[0], nogap_critical_keys[1]))[2:])

            # save tuples
            obj_tuples.append( # control 2x2
                gc.TupleSentenceData(
                    s_ab=extracted_item['what_gap_obj'], # filler, gap
                    s_xb=extracted_item['that_gap_obj'], # no filler, gap
                    s_ax=extracted_item['what_nogap_obj'], # filler, no gap
                    s_xx=extracted_item['that_nogap_obj'], # no filler, no gap
                )
            )
            wh_tuples.append( # wh- 2x2
                gc.TupleSentenceData(
                    s_ab=extracted_item['what_gap_wh'], # filler, gap
                    s_xb=extracted_item['that_gap_wh'], # no filler, gap
                    s_ax=extracted_item['what_nogap_wh'], # filler, no gap
                    s_xx=extracted_item['that_nogap_wh'], # no filler, no gap
                )
            )
            that_tuples.append( # that 2x2
                gc.TupleSentenceData(
                    s_ab=extracted_item['what_gap_that'], # filler, gap
                    s_xb=extracted_item['that_gap_that'], # no filler, gap
                    s_ax=extracted_item['what_nogap_that'], # filler, no gap
                    s_xx=extracted_item['that_nogap_that'], # no filler, no gap
                )
            )

    # calculate all surprisals
    all_tuples = [item for sublist in zip(obj_tuples, wh_tuples, that_tuples) for item in sublist]

    for item_tuple in all_tuples:
        surprisal.surprisal_effect_full_tuple(item_tuple, model, True)

    # save
    gc.corpus_to_json(all_tuples,
                      f"grammar_outputs/wilcox_replication/wilcox_cnp_islands_{model}.json")
    
    # load
    item_tuple_data = gc.corpus_from_json(
        f"grammar_outputs/wilcox_replication/wilcox_cnp_islands_{model}.json", is_tuples=True)
    # unpack items
    obj_tuples = item_tuple_data[::3]
    wh_tuples = item_tuple_data[1::3]
    that_tuples = item_tuple_data[2::3]

    # extract calculated surprisals from corpus
    # obj (control) wh-effects
    obj_gap_wh_effects = []
    obj_nogap_wh_effects = []
    for item in obj_tuples:
        # filler minus no filler
        obj_gap_wh_effects.append(item.s_ab.critical_surprisal - item.s_xb.critical_surprisal)
        obj_nogap_wh_effects.append(item.s_ax.critical_surprisal - item.s_xx.critical_surprisal)

    # wh-complement wh-effects
    wh_gap_wh_effects = []
    wh_nogap_wh_effects = []
    for item in wh_tuples:
        # filler minus no filler
        wh_gap_wh_effects.append(item.s_ab.critical_surprisal - item.s_xb.critical_surprisal)
        wh_nogap_wh_effects.append(item.s_ax.critical_surprisal - item.s_xx.critical_surprisal)

    # that-complement wh-effects
    that_gap_wh_effects = []
    that_nogap_wh_effects = []
    for item in that_tuples:
        # filler minus no filler
        that_gap_wh_effects.append(item.s_ab.critical_surprisal - item.s_xb.critical_surprisal)
        that_nogap_wh_effects.append(item.s_ax.critical_surprisal - item.s_xx.critical_surprisal)

    # average (according to user-chosen type)
    average_func = None

    if average_type == 'mean': average_func = np.mean
    elif average_type == 'median': average_type = np.median
    
    # plot
    plt.bar(
        ("cont. +gap", "cont. -gap", "th +gap", "th -gap", "wh +gap", "wh -gap"),
        (
            average_func(obj_gap_wh_effects),
            average_func(obj_nogap_wh_effects),
            average_func(that_gap_wh_effects),
            average_func(that_nogap_wh_effects),
            average_func(wh_gap_wh_effects),
            average_func(wh_nogap_wh_effects),
        ),
        color=['lightblue', 'lightcoral']
    )
    plt.title(f"Complex NP Islands Filler-Gap Dependency ({model})")
    plt.ylim(-5, 5)


def wilcox_basic_licensing(file_path: str,
                       nogap_critical_keys: tuple[str], gap_critical_keys: tuple[str],
                       gap_location: str, average_type: str = 'mean'):
    
    wh_tuples = []
    with open(f'wilcox_csv/{file_path}') as file:
        # convert CSV to list of dictionaries [str : str]
        csv_reader = list(csv.DictReader(file))

        items = [line.get('item') for line in csv_reader]
        # remove duplicates w/out losing ordering
        items = [i for n, i in enumerate(items) if i not in items[:n]]
        # is # of unique sets 

        for i in items:  # works for 4-way comparisons, ex: +/-filler, +/-gap
            sentence_datas = {}
            entire_item = [d for d in csv_reader if d.get('item') == i]
            for sentence in entire_item:
                # [2:] to skip item and condition values. ex: 1,what_nogap
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

    # save
    gc.corpus_to_json(wh_tuples,
                      f"grammar_outputs/wilcox_replication/wilcox_{gap_location}_wh_{model}.json")

    # load
    gap_tuple_data = gc.corpus_from_json(
        f"grammar_outputs/wilcox_replication/wilcox_{gap_location}_wh_{model}.json", is_tuples=True)

    # extract calculated surprisals from corpus
    gap_wh_effects = []
    nogap_wh_effects = []
    for sentence_tuple in gap_tuple_data:
        # (+gap wh-effect)
        gap_wh_effects.append(  # (wh, ___) - (that, ___)
            sentence_tuple.s_ab.critical_surprisal - sentence_tuple.s_xb.critical_surprisal)
        # (-gap wh-effect)
        nogap_wh_effects.append(  # (wh, no gap) - (that, no gap)
            sentence_tuple.s_ax.critical_surprisal - sentence_tuple.s_xx.critical_surprisal)

    if average_type == 'mean':
        avg_gap_wh_effect = np.mean(gap_wh_effects)
        avg_nogap_wh_effect = np.mean(nogap_wh_effects)
    elif average_type == 'median':
        avg_gap_wh_effect = np.median(gap_wh_effects)
        avg_nogap_wh_effect = np.median(nogap_wh_effects)
    else: # should not occur
        avg_gap_wh_effect = 0
        avg_nogap_wh_effect = 0
        

    # plot
    plt.bar(
        (
            f"{gap_location} +gap wh-effect",
            f"{gap_location} -gap wh-effect"
        ),
        (
            avg_gap_wh_effect,
            avg_nogap_wh_effect
        ),
        color=['lightblue', 'lightcoral']
    )
    plt.ylim(-1, 1)

"""
# subject
wilcox_basic_licensing('basic_subject.csv',  # where to load
                   ('np1', 'np1'), ('verb', 'verb'),  # where to extract
                   "subject", 'mean')  # title & average type

# object
wilcox_basic_licensing('basic_object.csv',  # where to load
                   ('np2', 'np2'), ('prep', 'np3'),  # where to extract
                   "object", 'mean')  # title & average type

# prepositional phrase
wilcox_basic_licensing('basic_pp.csv',  # where to load
                   ('np3', 'np3'), ('end', 'end'),  # where to extract
                   "pp", 'mean')  # title & average type
"""


wilcox_cnp_licensing('islands_cnp.csv', 
                     ('rc_obj', 'rc_obj'), ('continuation', 'continuation'),
                     'mean')

                     
plt.show()
