import argparse
from aitextgen import aitextgen, utils
from aitextgen.TokenDataset import TokenDataset

from tokenizer import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", help = "Path to vocab file")
    parser.add_argument("--train", help = "Path to training data")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.vocab)
    gpt2 = aitextgen(config = utils.GPT2Config())
    gpt2.tokenizer = tokenizer
    train_data = TokenDataset(file_path = args.train, tokenizer = tokenizer)
    # validation_data = TokenDataset(file_path = args.valid, tokenizer = tokenizer)
    train(gpt2, train_data)

def load_tokenizer(vocab : str) -> GRNNTokenizer:
    vocab = process_vocab_file(vocab, eos_token="<eos>", unk_token="<unk>")
    tokenizer = GRNNTokenizer(vocab)
    tokenizer.add_special_tokens({
        "unk_token": "<unk>",
        "eos_token": "<eos>"
    })
    print("Loaded Tokenizer")
    return tokenizer

def train(model : aitextgen, training_data : TokenDataset):
    # runs for one epoch (for now)
    print("Training model")
    batch_size = 200
    grnn_steps_per_epoch = 18541
    model.train(training_data, batch_size = batch_size, num_steps = batch_size * grnn_steps_per_epoch,
                 save_every = 2000, model_folder = "retrained_gpt2")

if __name__ == "__main__":
    main()
