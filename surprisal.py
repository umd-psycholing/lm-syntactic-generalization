import torch
import torch.nn.functional as F
import sys
from generate_corpora import SentenceData, TupleSentenceData

# sort of emulation of conditional compilation; this is an ugly temporary solution
# until we either find a way to generate the GRNN model w/ a version of torch
# compatible with minicons (torch>=2.0.0) or or replicate the functionality of
# minicons w/ version compatible w/ GRNN model (torch < 1.8.0).

if torch.__version__ >= "2.0":  # gpt2 (minicons)
    from minicons import scorer

    gpt2_model = scorer.IncrementalLMScorer(model="gpt2")

    # single sentence surprisal for gpt2
    def gpt2_surprisal(sentence: str) -> list[tuple[str, float]]:
        results = gpt2_model.token_score(
            batch=sentence, surprisal=True, base_two=True)
        # return first result since we are only doing one sentence at a time
        return results[0]
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

    # emulate GPT2 auto-tokenization
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

    not_in_vocab = []  # don't print unknown word over and over

    # convert word to index in embedding matrix
    def indexify(word, vocab):
        """ Convert word to an index into the embedding matrix """
        if word not in vocab.word2idx:
            # print non-vocabulary words only once
            if word not in not_in_vocab:
                not_in_vocab.append(word)
                print("Warning: {} not in vocab".format(word))
        # return index of word if its known, otherwise index of <unk> (unknown)
        return vocab.word2idx[word] if word in vocab.word2idx else vocab.word2idx["<unk>"]

    # NOTE: currently unused
    def align_surprisal(token_surprisals: list[tuple[str, float]], sentence: str):
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
    # ====== #

    sys.path.insert(
        0, "./colorlessgreenRNNs/src/language_models")
    # set up model
    torch.nn.Module.dump_patches = False
    lstm_vocab = load_vocab(
        "./colorlessgreenRNNs/src/data/lm")
    model, grnn = load_rnn(
        "./colorlessgreenRNNs/src/models/hidden650_batch128_dropout0.2_lr20.0.pt")

    # single sentence surprisal for gpt2
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
    (s_fg, s_xg, s_fx, s_xx) = (
        sentence_tuple.s_fg,
        sentence_tuple.s_xg,
        sentence_tuple.s_fx,
        sentence_tuple.s_xx
    )

    # generate each sentence's surprisal
    s_fg_surprisal = critical_surprisal_from_sentence(
        sentence=s_fg, model_to_use=model, update_class_field=update_class_fields)
    s_xg_surprisal = critical_surprisal_from_sentence(
        sentence=s_xg, model_to_use=model, update_class_field=update_class_fields)
    s_fx_surprisal = critical_surprisal_from_sentence(
        sentence=s_fx, model_to_use=model, update_class_field=update_class_fields)
    s_xx_surprisal = critical_surprisal_from_sentence(
        sentence=s_xx, model_to_use=model, update_class_field=update_class_fields)

    # defined externally since it may be calculated w/out re-calculating surprisal
    return compute_surprisal_effect_from_surprisals(s_fg_surprisal, s_xg_surprisal,
                                                    s_fx_surprisal, s_xx_surprisal)


# only implemented for model="gpt2", "grnn"
def critical_surprisal_from_sentence(sentence: SentenceData, model_to_use: str, update_class_field: bool = False):
    critical_text = sentence.critical_token

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
    critical_surprisal = None
    for token, surprisal_result in surprisal_info:
        if token == critical_text:
            critical_surprisal = surprisal_result
    if critical_surprisal == None:
        raise TypeError("Critical not found in surprisal data")

    if update_class_field:
        sentence.critical_surprisal = critical_surprisal

    return critical_surprisal
