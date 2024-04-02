from tokenizer import * 
from transformers import GPT2LMHeadModel
import torch

MODEL_DIR = "scripts/trained_model"
VOCAB_PATH = "grnn_data/vocab.txt"
sentence = "In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>"

vocab = process_vocab_file(VOCAB_PATH, "<eos>", "<unk>")
tokenizer = GRNNTokenizer(vocab)
gpt2 = GPT2LMHeadModel.from_pretrained(f"{MODEL_DIR}/pytorch_model.bin", config = f"{MODEL_DIR}/config.json")
gpt2.eval()

input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
with torch.no_grad():
    outputs = gpt2(input_ids, labels=input_ids)
loss, _ = outputs[:2]
print(torch.exp(loss))