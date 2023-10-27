from minicons import scorer
from typing import List, Tuple
import sys
import torch  # torch >= 2.0.0 (for minicons)
import torch.nn.functional as F


##################
# GPT2           #
# (torch>=2.0.0) #
##################


gpt2_model = scorer.IncrementalLMScorer("gpt2")


def gpt2_surprisal(sentence):
    # returns [(token, score), (token, score), ...]
    results = gpt2_model.token_score(
        sentence, surprisal=True, base_two=True)[0]
    return results
