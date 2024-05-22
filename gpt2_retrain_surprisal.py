import os
import numpy as np
import torch
from torch.nn.functional import F
from transformers import GPT2LMHeadModel

from tokenizer import * 

ROOT = "/fs/clip-psych/sathvik/lm-syntactic-generalization"
MODEL_DIR = os.path.join(ROOT, "scripts/trained_model")
VOCAB_PATH = os.path.join(ROOT, "grnn_data/vocab.txt")

vocab = process_vocab_file(VOCAB_PATH, "<eos>", "<unk>")
tokenizer = GRNNTokenizer(vocab)
gpt2 = GPT2LMHeadModel.from_pretrained(f"{MODEL_DIR}/pytorch_model.bin", config = f"{MODEL_DIR}/config.json")

wh = 'I know what with gusto our uncle grabbed the food in front of the guests at the holiday party . <eos>'
that = 'I know that with gusto our uncle grabbed the food in front of the guests at the holiday party . <eos>'

def prepare_text(sentence):
    tokens = tokenizer.tokenize(sentence)
    for i in range(len(tokens)):
        if tokens[i] not in vocab:
            tokens[i] = "<unk>"
    return tokens, torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

def sentence_surprisal(sentence):
    tokens, model_input = prepare_text(sentence)
    with torch.no_grad():
        logits = gpt2(model_input).logits

    surprisals = -F.log_softmax(logits) / np.log(2.0)
    tokenwise_surprisals = [(tokens[i], surprisals[i][model_input[i]].item()) for i in np.arange(len(model_input))]
    return tokenwise_surprisals
