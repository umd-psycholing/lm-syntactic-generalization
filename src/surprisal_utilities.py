import numpy as np
import multiprocessing
import os
import matplotlib.pyplot as plt

from build_sentence_tuples import group_sentences


def delta_delta_from_group(tuple: dict[str, dict[str, str]], model):
    wh_gap = tuple['S_FG']['Sentence']
    that_gap = tuple['S_XG']['Sentence']
    gap_critical = tuple['S_FG']['Critical String']

    wh_nogap = tuple['S_FX']['Sentence']
    that_nogap = tuple['S_XX']['Sentence']
    nogap_critical = tuple['S_FX']['Critical String']

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
