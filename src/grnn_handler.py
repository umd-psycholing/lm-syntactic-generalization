from colorlessgreenRNNs.src.language_models.dictionary_corpus import Dictionary
from colorlessgreenRNNs.src.language_models.model import RNNModel
import sys
import torch
import torch.nn.functional as F

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


def get_grnn_surprisal(sentence):
    sentence = ["<eos>"] + tokenize(sentence)  # EOS prepend
    rnn_input = torch.LongTensor([indexify(w.lower()) for w in sentence])
    out, _ = grnn(rnn_input.view(-1, 1), model.init_hidden(1))
    output_flat = out.view(-1, len(vocab))
    return [-F.log_softmax(out[i], dim=-1).view(-1)[word_idx].item() for i, (word_idx, word)
            in enumerate(zip(rnn_input, sentence))][1:-1]
