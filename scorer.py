import matplotlib.pyplot as plt
from minicons import scorer

wh_gap = "I know what John looked for yesterday and will devour tomorrow"
that_gap = "I know that John looked for food yesterday and will devour tomorrow."
gap_critical = "tomorrow"

wh_nogap = "I know what John looked for yesterday and will devour it tomorrow"
that_nogap = "I know that John looked for food yesterday and will devour it tomorrow."
nogap_critical = "it"

model = scorer.IncrementalLMScorer("gpt2")


def delta_delta_from_tuple(model, wh_gap, wh_nogap, that_gap, that_nogap, gap_critical, nogap_critical):
    # delta -filler - delta +filler
    plus_filler = compute_delta(
        model, that_gap, gap_critical, that_nogap, nogap_critical)
    minus_filler = compute_delta(
        model, wh_gap, gap_critical, wh_nogap, nogap_critical)

    return plus_filler - minus_filler


def compute_delta(model, gap_sentence: str, gap_critical: str, nogap_sentence: str, nogap_critical: str):
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


print(delta_delta_from_tuple(
    model,
    wh_gap, wh_nogap,
    that_gap, that_nogap,
    gap_critical, nogap_critical
))

# new issue is building tuples...
# when generating tuples, start by deciding on a permutation of all the reserved types. THEN, the constructions should be deterministic, and it will produce four sentences for you.
# do that for each permutation of reserved types and you should be in business!
