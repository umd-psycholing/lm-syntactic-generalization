from colorlessgreenRNNs.src.language_models.dictionary_corpus import Dictionary
from colorlessgreenRNNs.src.language_models.model import RNNModel
from typing import List, Tuple
import sys
# torch<=1.7.X (for colorlessgreenRNNs' pretrained model, if I can retrain GRNN w/ a newer torch I shall)
import torch
import torch.nn.functional as F


#################
# GRNN          #
# (torch<1.8.0) #
#################

sys.path.insert(0, ".")


# load GRNN
def load_rnn(model_path):
    # this assumes we're using the CPU, which should be fine for inference
    # we can change the settings to allow GPU inference if needed
    model = torch.load(model_path, map_location=torch.device("cpu"))
    grnn = RNNModel(model.rnn_type, model.encoder.num_embeddings,
                    model.nhid, model.nhid, model.nlayers, 0.2, False)
    grnn.load_state_dict(model.state_dict())
    grnn.eval()
    return model, grnn


# calculate surprisal
def grnn_surprisal(model: RNNModel, grnn: RNNModel, vocab: Dictionary, sentence):
    sentence = ["<eos>"] + tokenize(sentence)  # EOS prepend
    rnn_input = torch.LongTensor(
        [indexify(w.lower(), vocab) for w in sentence])
    out, _ = grnn(rnn_input.view(-1, 1), model.init_hidden(1))
    return [-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() for i, (word_idx, word)
            in enumerate(zip(rnn_input, sentence))][1:-1]


# vocab comes from colorlessgreenRNNs/src/data/lm/vocab.txt (where gulordova repo references it)
def load_vocab(vocab_path):
    # loads vocabulary for RNN model
    # the path must be a directory
    return Dictionary(vocab_path)


not_in_vocab = []


def indexify(word, vocab):
    """ Convert word to an index into the embedding matrix """
    if word not in vocab.word2idx:
        # print non-vocabulary words only once
        if word not in not_in_vocab:
            not_in_vocab.append(word)
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


# NOTE: currently unused, should ask sathvik abt the purpose of this
def align_surprisal(token_surprisals: List[Tuple[str, float]], sentence: str):
    # this is used to tokenize RNN input but if we're going to compare GPT outputs we might as well use the same technique
    words = tokenize(sentence)
    token_index = 0
    word_index = 0
    word_level_surprisal = []  # list of word, surprisal tuples
    while token_index < len(token_surprisals):
        current_word = words[word_index]
        current_token, current_surprisal = token_surprisals[
            token_index][0], token_surprisals[token_index][1]
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


# makes sure torch just prints its complaints about the source change, and doesn't spit out .patch files
torch.nn.Module.dump_patches = False

sys.path.insert(
    0, "./src/colorlessgreenRNNs/src/language_models")
lstm_vocab = load_vocab(
    "./src/colorlessgreenRNNs/src/language_models/../data/lm")
model, grnn = load_rnn(
    "./src/colorlessgreenRNNs/src/language_models/../models/hidden650_batch128_dropout0.2_lr20.0.pt")


def grnn_surprisal(sentence: str, model: RNNModel = model, grnn: RNNModel = grnn, vocab: Dictionary = lstm_vocab):
    sentence = ["<eos>"] + tokenize(sentence)  # EOS prepend
    rnn_input = torch.LongTensor(
        # [indexify(w.lower(), vocab) for w in sentence]) # lowercase names are not in vocab!
        [indexify(w, vocab) for w in sentence])
    out, _ = grnn(rnn_input.view(-1, 1), model.init_hidden(1))
    surprisals = [-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() for i, (word_idx, word)
                  in enumerate(zip(rnn_input, sentence))]
    # [1:] skips "<eos>" and corresponding surprisal
    return (list(zip(sentence[1:], surprisals[1:])))
