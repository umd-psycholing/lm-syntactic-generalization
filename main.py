import surprisal
import argparse
from typing import List, Dict

from nltk.grammar import CFG

from grammars import get_grammar
import generate_corpora as gc
import sentence_tuples

# example: python main.py --model gpt2 --sentence_type cleft -island -training --save_to XXX.json (cleft +island,+training GRNN)
# example: python main.py --sentence_type cleft --save_to XXX.json (cleft -island,-training CORPUS)
# example: python main.py --sentence_type tough -training -unformatted --save_to XXX.txt (tough -island,+training SENTENCE_LIST)


# used to just get sentence_lists
def write_unformatted_to_file(objects, filename):
    with open(filename, 'w') as f:
        for obj in objects:
            f.write(str(obj) + '\n')


### MAIN ###


def main():
    # trained models added here
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['grnn', 'gpt2'],
                        help='choose between grnn or gpt2. Do not provide a model for just the formatted corpus. ')
    parser.add_argument("--sentence_type", choices=['cleft', 'intro_topic', 'no_intro_topic', 'tough'],
                        help='construction type for grammar')
    parser.add_argument("-island", action='store_true', default=False,
                        help='Enable for island grammar')
    parser.add_argument("-training", action='store_true', default=False,
                        help='Enable for training grammar')
    parser.add_argument("-unformatted", action='store_true', default=False,
                        help='Enable to just generate a list of all sentences produced by gramamr')
    parser.add_argument("--save_to", default="surprisal.json",
                        help='Where to save file')
    args = parser.parse_args()

    # get grammar
    grammar = CFG.fromstring(get_grammar(type=args.sentence_type,
                                         island=args.island, training=args.training))
    print(f"grammar [{args.sentence_type}, {'+' if args.island else '-'}island {'+' if args.training else '-'}training] generated")

    # generate sentences
    if args.unformatted:
        corpus = gc.generate_all_sentences_from_grammar(grammar=grammar)
        print(f"{len(corpus)} sentences generated")
    else:
        corpus = gc.generate_all_sentence_tuples_from_grammar(grammar=grammar)
        print(f"{len(corpus)} tuples generated")

    # calculate surprisal if model is available
    if hasattr(args, 'model') and not args.unformatted:
        model = args.model
        corpus = surprisal.surprisal_total_corpus(
            corpus=corpus, model=model)
        print(f"surprisals calculated for {model}")

    # save
    if not args.unformatted:
        sentence_tuples.corpus_to_json(
            input_data=corpus, filename=args.save_to)
        print(f"corpus saved to {args.save_to}")
    else:
        write_unformatted_to_file(objects=corpus, filename=args.save_to)
        print(f"sentences saved to {args.save_to}")


if __name__ == "__main__":
    main()
