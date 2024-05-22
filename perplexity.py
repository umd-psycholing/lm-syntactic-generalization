import os
from tokenizer import * 
from transformers import GPT2LMHeadModel
import torch
from tqdm import tqdm

ROOT = "/fs/clip-psych/sathvik/lm-syntactic-generalization"
MODEL_DIR = os.path.join(ROOT, "scripts/trained_model")
VOCAB_PATH = os.path.join(ROOT, "grnn_data/vocab.txt")
TEST_DATA = os.path.join(ROOT, "grnn_data/test.txt")

device = torch.device("cuda")
vocab = process_vocab_file(VOCAB_PATH, "<eos>", "<unk>")
tokenizer = GRNNTokenizer(vocab)
gpt2 = GPT2LMHeadModel.from_pretrained(f"{MODEL_DIR}/pytorch_model.bin", config = f"{MODEL_DIR}/config.json")
gpt2.eval()
gpt2.to("cuda")

with open(TEST_DATA) as f:
    test_sentences = f.readlines()

total_loss = 0
total_tokens = 0
print_interval = 0
total_sentences = 0

def sentence_level_perplexity(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence.replace("\n", ""))).unsqueeze(0).to("cuda")
    with torch.no_grad():
        outputs = gpt2(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return loss, input_ids.size(dim = 1)
for sentence in tqdm(test_sentences):
    loss, tokens = sentence_level_perplexity(sentence)
    total_loss += loss
    total_tokens += tokens
    total_sentences += 1
    print_interval += 1
    if print_interval == 50:
        print("Test set perplexity", float(torch.exp(total_loss / total_sentences)))
        print_interval = 0

print("Test set perplexity", float(torch.exp(total_loss / total_sentences)))