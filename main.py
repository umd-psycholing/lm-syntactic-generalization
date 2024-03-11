import surprisal
import argparse
from typing import List, Dict

from nltk.grammar import CFG

from grammars import get_grammar
import generate_corpora as gc
import sentence_tuples

# example: python main.py --model gpt2 --sentence_type cleft -island -training --save_to XXX.json


### MAIN ###
def main():
    # trained models added here
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['grnn', 'gpt2'],
                        help='choose between grnn or gpt2')
    parser.add_argument("--sentence_type", choices=['cleft', 'intro_topic', 'no_intro_topic', 'tough'],
                        help='construction type for grammar')
    parser.add_argument("-island", action='store_true', default=False,
                        help='Enable for island grammar')
    parser.add_argument("-training", action='store_true', default=False,
                        help='Enable for training grammar')
    parser.add_argument("--save_to", default="surprisal.json",
                        help='Where to save file')
    args = parser.parse_args()

    # get grammar
    grammar = CFG.fromstring(get_grammar(type=args.sentence_type,
                                         island=args.island, training=args.training))
    print(f"grammar [{args.sentence_type}, {'+' if args.island else '-'}island {'+' if args.training else '-'}training] generated")

    # generate sentences
    corpus = gc.generate_all_sentence_tuples_from_grammar(grammar=grammar)
    print(f"{len(corpus)} tuples generated")

    # calculate surprisal
    corpus_surprisal = surprisal.surprisal_total_corpus(
        corpus=corpus, model=args.model)
    print(f"surprisals calculated for {args.model}")

    # save
    sentence_tuples.corpus_to_json(
        input_data=corpus_surprisal, filename=args.save_to)
    print(f"surprisal-containing corpus saved to {args.save_to}")


if __name__ == "__main__":
    main()
