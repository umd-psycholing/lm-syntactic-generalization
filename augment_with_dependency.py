import argparse
import os

from generate_corpora import corpus_from_json
from surprisal import grnn_tokenize

RANDOM_SEED = 29

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help = "path to directory with data for LM")
    parser.add_argument("--dependency_name")
    parser.add_argument("--augmenting_data", help = "path to directory with dependency")
    args = parser.parse_args()
    sentences = sentence_list(args.data_dir)
    write_sentence_to_file(sentences, args.data_dir, args.dependency_name)


def sentence_list(data_path):
    sentence_tuples = corpus_from_json(data_path, is_tuples=True)
    return [grnn_tokenize(tuple.processed_tokens) for tuple in sentence_tuples]

def write_sentence_to_file(sentence_list, data_dir, dependency_name):
    if dependency_name not in os.listdir(data_dir):
        os.mkdir(dependency_name)
    train_lines = open(os.path.join(data_dir, "train.txt"), "r").readlines()
    valid_lines = open(os.path.join(os.path.join(data_dir, "valid.txt")), "r").readlines()

    cutoff = len(sentence_list) / 9 # Gulordava et al leave 1/9 of the training data as validation data 
    valid_sentences = sentence_list[:cutoff]
    train_sentences = sentence_list[cutoff:]

    for sentence in sentence_list:
        " ".join(sentence) + " <eos>"
    # TODO write this to a file
    return

if __name__ == "__main__":
    main()