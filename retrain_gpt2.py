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
    gpt2 = load_model(tokenizer)
    train(gpt2, args.train)

def load_model(tokenizer : GRNNTokenizer) -> aitextgen:
    model_config = utils.GPT2Config()
    model_config.vocab_size = tokenizer.vocab_size
    gpt2 = aitextgen(config = utils.GPT2Config())
    gpt2.tokenizer = tokenizer
    return gpt2

def load_tokenizer(vocab : str) -> GRNNTokenizer:
    vocab = process_vocab_file(vocab, eos_token="<eos>", unk_token="<unk>")
    tokenizer = GRNNTokenizer(vocab)
    tokenizer.add_special_tokens({
        "unk_token": "<unk>",
        "eos_token": "<eos>"
    })
    print("Loaded Tokenizer")
    return tokenizer

def train(model : aitextgen, train_path : str):
    # runs for one epoch (for now)
    # if we use aitextgen for augmentation, we should shuffle the dataset
    # maybe test val accuracy after x steps?
    print("Training model")
    batch_size = 10
    with open(train_path) as f:
        sentence_count = len(f.readlines())
    model.train(train_path, line_by_line = True, batch_size = batch_size, num_steps = round(sentence_count / batch_size),
                 save_every = 10000, num_workers = 2, gradient_accumulation_steps = 8, generate_every = -1,
                 model_folder = "retrained_gpt2")

if __name__ == "__main__":
    main()
