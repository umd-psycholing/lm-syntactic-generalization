import argparse
import os
import numpy as np
import pandas as pd

from surprisal import grnn_tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help = "path to directory with data for LM")
    parser.add_argument("--dependency_name")
    parser.add_argument("--augmenting_data", help = "path to directory with dependency")
    args = parser.parse_args()
    print("Reading and tokenizing sentences")
    sentences = sentence_list(args.augmenting_data)
    write_sentence_to_file(sentences, args.data_dir, args.dependency_name)


def sentence_list(data_path):
    """
    this extracts the grammatical sentences and tokenizes them.
    we need to make sure there's an equal amount of with/without filler & gap sentences 
    when we split the data into training and validation 
    """
    sentences = pd.read_csv(data_path)
    grammatical_sentences = sentences[sentences['grammatical']]
    return grammatical_sentences['sentence'].apply(grnn_tokenize).values

def write_sentence_to_file(sentence_list, data_dir, dependency_name):
    if dependency_name not in os.listdir(data_dir):
        os.mkdir(os.path.join(data_dir, dependency_name))
    train_lines = open(os.path.join(data_dir, "train.txt"), "r").readlines()
    valid_lines = open(os.path.join(os.path.join(data_dir, "valid.txt")), "r").readlines()

    cutoff = int(np.round(len(sentence_list) / 9)) # Gulordava et al leave 1/9 of the training data as validation data 
    valid_sentences = sentence_list[:cutoff]
    train_sentences = sentence_list[cutoff:]

    train_lines += [convert_tokens(sentence) for sentence in train_sentences] 
    valid_lines += [convert_tokens(sentence) for sentence in valid_sentences]

    train_path = os.path.join(data_dir, dependency_name, "train.txt")
    valid_path = os.path.join(data_dir, dependency_name, "valid.txt")

    print("Writing training file")
    with open(train_path, "w") as file:
        file.writelines(train_lines)
    
    print("Writing validation file")
    with open(valid_path, "w") as file:
        file.writelines(valid_lines)

def convert_tokens(sentence):
    return " ".join(sentence + ["."] + ['<eos>']) + "\n"

if __name__ == "__main__":
    main()