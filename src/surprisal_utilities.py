from minicons import scorer


def compute_delta_gap(model_func: callable, gap_sentence: str, gap_critical: str, nogap_sentence: str, nogap_critical: str):
    gap_surprisals = model_func(gap_sentence)
    gap_critical_surprisal = next((token[1] for token in gap_surprisals
                                   if token[0] == gap_critical), 0)

    nogap_surprisals = model_func(nogap_sentence)
    nogap_critical_surprisal = next((token[1] for token in nogap_surprisals
                                     if token[0] == nogap_critical), 0)

    return gap_critical_surprisal - nogap_critical_surprisal


gpt2_model = scorer.IncrementalLMScorer("gpt2")


def gpt2_surprisal(sentence):
    # returns [(token, score), (token, score), ...]
    results = gpt2_model.token_score(
        sentence, surprisal=True, base_two=True)[0]
    return results


def grnn_surprisal(sentence):
    return grnn_surprisal(sentence)
