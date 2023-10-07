from typing import List, Tuple

import torch
import torch.nn.functional as F

from colorlessgreenRNNs.src.language_models.model import RNNModel
from colorlessgreenRNNs.src.language_models.dictionary_corpus import Dictionary

def load_rnn(model_path):
    # this assumes we're using the CPU, which should be fine for inference
    # we can change the settings to allow GPU inference if needed
    model = torch.load(model_path, map_location = torch.device("cpu"))
    grnn = RNNModel(model.rnn_type, model.encoder.num_embeddings, 
                        model.nhid, model.nhid, model.nlayers, 0.2, False)
    grnn.load_state_dict(model.state_dict())
    grnn.eval()
    return model, grnn

def grnn_surprisal(model: RNNModel, grnn: RNNModel, vocab: Dictionary, sentence):
    sentence = ["<eos>"] + tokenize(sentence) # EOS prepend
    rnn_input = torch.LongTensor([indexify(w.lower(), vocab) for w in sentence])
    out, _ = grnn(rnn_input.view(-1, 1), model.init_hidden(1))
    return [-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() for i, (word_idx, word)
 in enumerate(zip(rnn_input, sentence))][1:-1]

def load_vocab(vocab_path):
    # loads vocabulary for RNN model
    # the path must be a directory
    return Dictionary(vocab_path)

def indexify(word, vocab):
    """ Convert word to an index into the embedding matrix """
    if word not in vocab.word2idx:
        print("Warning: {} not in vocab".format(word))
    return vocab.word2idx[word] if word in vocab.word2idx else vocab.word2idx["<unk>"]

def tokenize(sent):
    sent = sent.strip()
    if sent == "": return []
    # respect commas as a token
    sent = " ,".join(sent.split(","))
    # same w/ EOS punctuation (but not . in abbreviations)
    if sent[-1] in  [".", "?", "!"]:
        sent = sent[:-1] + " " + sent[-1]
    if ("." in sent) & (sent[-1] != "."):
        print(sent)
    # split on 's
    sent = " 's".join(sent.split("'s"))
    # split on n't
    sent = " n't".join(sent.split("n't"))
    return sent.split()

def align_surprisal(token_surprisals: List[Tuple[str, float]], sentence: str):
    words = tokenize(sentence) # this is used to tokenize RNN input but if we're going to compare GPT outputs we might as well use the same technique
    token_index = 0
    word_index = 0
    word_level_surprisal = [] # list of word, surprisal tuples
    while token_index < len(token_surprisals):
        current_word = words[word_index]
        current_token, current_surprisal = token_surprisals[token_index][0], token_surprisals[token_index][1]
        mismatch = current_word != current_token
        while mismatch:
            token_index += 1
            current_token += token_surprisals[token_index][0]
            current_surprisal += token_surprisals[token_index][1]
            mismatch = current_token != current_word
        word_level_surprisal.append((current_word, current_surprisal))
        token_index += 1
        word_index += 1
    return word_level_surprisal