from multiprocessing import Pool
from functools import partial  # used to define default values
import numpy as np
import torch
import torch.nn.functional as F
import sys
from generate_corpora import SentenceData, TupleSentenceData

from typing import Iterable

# sort of emulation of conditional compilation; this is an ugly temporary solution
# until we either find a way to generate the GRNN model w/ a version of torch
# compatible with minicons (torch>=2.0.0) or or replicate the functionality of
# minicons w/ version compatible w/ GRNN model (torch < 1.8.0).

if torch.__version__ >= "2.0":  # gpt2 (minicons)
    from minicons import scorer

    gpt2_model = scorer.IncrementalLMScorer(model="gpt2")

    # single sentence surprisal for gpt2
    def gpt2_surprisal(sentence: str) -> list[tuple[str, float]]:
        BOS_TOKEN = "<|endoftext|> "
        EOS_TOKEN = " <|endoftext|>"
        results = gpt2_model.token_score(
            batch=BOS_TOKEN + sentence + EOS_TOKEN, surprisal=True, base_two=True)  # surprisal=True just flips signs relative to surprisal=False (log-odds)
        # return first result since we are only doing one sentence at a time
        return align_surprisal(results[0], BOS_TOKEN + sentence + EOS_TOKEN)

elif torch.__version__ <= "1.8":  # grnn (colorlessgreenRNNs)
    from colorlessgreenRNNs.src.language_models.dictionary_corpus import Dictionary
    from colorlessgreenRNNs.src.language_models.model import RNNModel

    # ===Sathvik-created helping functions for GRNN=== #
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

    # vocab comes from ./colorlessgreenRNNs/src/data/lm/vocab.txt (where gulordova repo references it)
    def load_vocab(vocab_path):
        # loads vocabulary for RNN model
        # the path must be a directory
        return Dictionary(vocab_path)

    # GRNN model expects this sort of tokenization for input (split -'s is critical)
    def grnn_tokenize(sent):
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

    not_in_vocab = []  # don't print unknown word over and over

    # convert word to index in embedding matrix
    def indexify(word, vocab):
        """ Convert word to an index into the embedding matrix """
        if word not in vocab.word2idx:
            # print non-vocabulary words only once
            if word not in not_in_vocab:
                not_in_vocab.append(word)
                print(f"Warning: {word} not in vocab")
        # return index of word if its known, otherwise index of <unk> (unknown)
        return vocab.word2idx[word] if word in vocab.word2idx else vocab.word2idx['<unk>']

    # load model, vocab once
    sys.path.insert(
        0, "./colorlessgreenRNNs/src/language_models")
    # set up model
    torch.nn.Module.dump_patches = False
    lstm_vocab = load_vocab("./colorlessgreenRNNs/src/data/lm/")
    model, grnn = load_rnn(
        "./colorlessgreenRNNs/src/models/model_clefting.pt")

    # single sentence surprisal for gpt2
    def grnn_surprisal(sentence: str, model: RNNModel = model, grnn: RNNModel = grnn, vocab: Dictionary = lstm_vocab):
        # EOS prepend + 's split
        # tokens = ["<eos>"] + grnn_tokenize(sentence)
        with torch.no_grad():
            tokens = ["<eos>"] + grnn_tokenize(sentence)
            rnn_input = torch.LongTensor(
                # [indexify(w.lower(), vocab) for w in sentence]) # lowercase names are not in vocab!
                [indexify(w, vocab) for w in tokens])
            out, _ = grnn(rnn_input.view(-1, 1), model.init_hidden(1))
            surprisals = [-F.log_softmax(out[i - 1], dim=-1).view(-1)[word_idx].item()/np.log(2.0) for i, (word_idx, word)
                          in enumerate(zip(rnn_input, tokens))]
            surprisals = list(zip(tokens, surprisals))  # zip tokens in w/ it
            return align_surprisal(surprisals, "<eos> " + sentence)


def align_surprisal(token_surprisals: list[tuple[str, float]], sentence: str):
    words = sentence.split(" ")
    token_index = 0
    word_index = 0
    word_level_surprisal = []  # list of word, surprisal tuples
    while token_index < len(token_surprisals):
        current_word = words[word_index]
        current_token, current_surprisal = token_surprisals[token_index]
        # token does not match, alignment must be adjusted
        mismatch = (current_token != current_word)

        while mismatch:
            token_index += 1
            current_token += token_surprisals[token_index][0]
            current_surprisal += token_surprisals[token_index][1]
            mismatch = current_token != current_word
        word_level_surprisal.append((current_word, current_surprisal))
        token_index += 1
        word_index += 1
    return word_level_surprisal


# meant for standard 4-way comparison, somewhat niche. maybe should be removed
def compute_surprisal_effect_from_surprisals(s_fg_surprisal: float, s_xg_surprisal: float,
                                             s_fx_surprisal: float, s_xx_surprisal: float):
    # how much more surprising is a gap, assuming a filler? (should be very low/negative)
    delta_plus_filler = (s_fg_surprisal - s_fx_surprisal)
    # how much better is a gap, assuming no filler? (should be very high)
    delta_minus_filler = (s_xg_surprisal - s_xx_surprisal)

    # we expect to see a positive value here if the model is doing a good job!
    return delta_minus_filler - delta_plus_filler


# wrappers for model-specific, single-sentence surprisal functions
def surprisal_effect_full_tuple(sentence_tuple: TupleSentenceData, model: str, update_class_fields: bool = False):
    # 'unpack' tuple
    (s_ab, s_xb, s_ax, s_xx) = (
        sentence_tuple.s_ab,
        sentence_tuple.s_xb,
        sentence_tuple.s_ax,
        sentence_tuple.s_xx
    )

    # generate each sentence's surprisal
    s_ab_surprisal = critical_surprisal_from_sentence(
        sentence=s_ab, model_to_use=model, update_class_field=update_class_fields)
    s_xb_surprisal = critical_surprisal_from_sentence(
        sentence=s_xb, model_to_use=model, update_class_field=update_class_fields)
    s_ax_surprisal = critical_surprisal_from_sentence(
        sentence=s_ax, model_to_use=model, update_class_field=update_class_fields)
    s_xx_surprisal = critical_surprisal_from_sentence(
        sentence=s_xx, model_to_use=model, update_class_field=update_class_fields)

    # return changed corpus
    return sentence_tuple


def surprisal_total_corpus(corpus: Iterable[TupleSentenceData], model: str):
    print("pooling")
    partial_func_surprisal_effect = partial(
        surprisal_effect_full_tuple, model=model)

    with Pool() as pool:
        results = pool.map(partial_func_surprisal_effect, corpus)

    return results


def _sum_surprisals(tokens_and_scores: list[tuple[str, float]], target_tokens: list[str]) -> float:
    # print(tokens_and_scores, target_tokens)
    total_score = 0.0
    current_sequence = []
    for tuple_word, score in tokens_and_scores:
        # print(current_sequence)

        # beginning of match is found
        if current_sequence:
            # next word is found
            if tuple_word == target_tokens[len(current_sequence)]:
                current_sequence.append(tuple_word)
                total_score += score
            else:  # non-matching word is found
                current_sequence = []
                total_score = 0

        # if sequence is empty and first target matches
        if not current_sequence and tuple_word == target_tokens[0]:
            current_sequence.append(tuple_word)
            total_score += score

        # total match is found
        if current_sequence == target_tokens:
            return total_score

    raise RuntimeError("Target tokens not found. (_sum_surprisals())")


# implemented for model="gpt2", "grnn"
def critical_surprisal_from_sentence(sentence: SentenceData, model_to_use: str, update_class_field: bool = False):
    critical_tokens = sentence.critical_tokens

    # calculate surprisal from indicated model
    if model_to_use == "gpt2":
        surprisal_info = gpt2_surprisal(
            sentence=str(sentence))
    elif model_to_use == "grnn":
        surprisal_info = grnn_surprisal(
            model=model, grnn=grnn, vocab=lstm_vocab, sentence=str(sentence)
        )
    else:
        raise ValueError(
            "Model not recognized. Valid models include: 'gpt2', 'grnn'")

    # get critical surprisal
    critical_surprisal = _sum_surprisals(surprisal_info, critical_tokens)

    """
    # this way of doing it over-counted repeat words. 
    # Ex: "I know the cat ate the bird", ("the", "bird") would add the surprisal of 'the', 'the', and 'bird'.

    critical_surprisal = 0
    for token, surprisal_result in surprisal_info:
        if token in critical_tokens:
            critical_surprisal += surprisal_result  # sum surprisal of each critical token
    """

    if update_class_field:
        sentence.critical_surprisal = critical_surprisal

    return critical_surprisal
