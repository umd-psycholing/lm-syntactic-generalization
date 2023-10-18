from minicons import scorer
# from grnn_handler import get_grnn_surprisal

gpt2_model = scorer.IncrementalLMScorer("gpt2")


def gpt2_surprisal(sentence):
    # returns [(token, score), (token, score), ...]
    results = gpt2_model.token_score(
        sentence, surprisal=True, base_two=True)[0]
    return results


def grnn_surprisal(sentence):
    pass
    # return surprisal.grnn_surprisal(sentence)
