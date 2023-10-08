from colorlessgreenRNNs.src.language_models.dictionary_corpus import Dictionary
from colorlessgreenRNNs.src.language_models.model import RNNModel
import torch.nn.functional as F
import torch
import sys
import numpy as np
import multiprocessing
import os
import matplotlib.pyplot as plt
from minicons import scorer

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


gpt2_model = scorer.IncrementalLMScorer("gpt2")


def gpt_surprisal(sentence):
    # returns [(token, score), (token, score), ...]
    results = gpt2_model.token_score(
        sentence, surprisal=True, base_two=True)[0]
    return results


def grnn_surprisal(sentence):
    pass


sys.path.insert(0, "./colorlessgreenRNNs/src/language_models")


model_file = "hidden650_batch128_dropout0.2_lr20.0.pt"
model = torch.load(model_file, map_location=torch.device("cpu"))
grnn = RNNModel(model.rnn_type, model.encoder.num_embeddings,
                model.nhid, model.nhid, model.nlayers, 0.2, False)
grnn.load_state_dict(model.state_dict())
grnn.eval()


vocab = Dictionary(".")  # takes in a directory
# from https://github.com/caplabnyu/sapbenchmark/blob/main/Surprisals/get_lstm.py


def indexify(word):
    """ Convert word to an index into the embedding matrix """
    if word not in vocab.word2idx:
        print("Warning: {} not in vocab".format(word))
    return vocab.word2idx[word] if word in vocab.word2idx else vocab.word2idx["<unk>"]


def tokenize(sent):
    sent = sent.strip()
    if sent == "":
        return []

    # respect commas as a token
    sent = " ,".join(sent.split(","))

    # same w/ EOS punctuation (but not . in abbreviations)
    if sent[-1] in [".", "?", "!"]:
        sent = sent[:-1] + " " + sent[-1]

    if ("." in sent) & (sent[-1] != "."):
        print(sent)

    # split on 's
    sent = " 's".join(sent.split("'s"))

    # split on n't
    sent = " n't".join(sent.split("n't"))

    return sent.split()


def grnn_surprisal(sentence):
    sentence = ["<eos>"] + tokenize(sentence)  # EOS prepend
    rnn_input = torch.LongTensor([indexify(w.lower()) for w in sentence])
    out, _ = grnn(rnn_input.view(-1, 1), model.init_hidden(1))
    output_flat = out.view(-1, len(vocab))
    return [-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() for i, (word_idx, word)
            in enumerate(zip(rnn_input, sentence))][1:-1]


print(grnn_surprisal("The cat meowed"))
