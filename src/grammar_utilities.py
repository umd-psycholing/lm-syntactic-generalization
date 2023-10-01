import time
import csv

"""
_summary_
"""


def _expand_first_non_terminal(input_sentence: list, grammar: dict[str]) -> list[list]:
    # element which will be expanded, either string non-terminal or array
    for element in input_sentence:
        if isinstance(element, list):
            first_non_terminal = element
            break
        elif element in grammar.keys():
            first_non_terminal = element
            break

    if isinstance(element, str):
        return _expand_string_element(input_sentence, first_non_terminal, grammar.get(first_non_terminal))

    # grammar is not required since all replacement info is within arrays themselves
    elif isinstance(element, list):
        return _expand_array_element(input_sentence, first_non_terminal)


def _expand_string_element(input_sentence: list, replacement_non_terminal: str, replacement_expansions: list) -> list[list]:
    resulting_sentences = []
    for expansion in replacement_expansions:
        expansion_sentence = []
        for element in input_sentence:
            if element != replacement_non_terminal:
                expansion_sentence.append(element)
            else:
                expansion_sentence.extend(expansion)
        resulting_sentences.append(expansion_sentence)
    return resulting_sentences


def _expand_array_element(input_sentence: list, replacement_expansions: list) -> list[list]:
    resulting_sentences = []
    for expansion in replacement_expansions:
        expansion_sentence = []
        for element in input_sentence:
            if element != replacement_expansions:
                expansion_sentence.append(element)
            else:
                expansion_sentence.append(expansion)
        resulting_sentences.append(expansion_sentence)
    return resulting_sentences


def _all_leaves(input_sentence: list, grammar: dict[str]) -> bool:
    for element in input_sentence:
        # OR--must be broken up
        if isinstance(element, list):
            return False
        # non-terminal
        if element in grammar.keys():
            return False
    # all elements must be leaves
    return True


def generate_sentences(grammar, starting_symbol):
    # queue starts with base node
    # until queue is empty, remove first sentence
    # this sentence's first non-terminal node is expanded and sentence is added to the queue
    # if sentence has no non-terminal nodes it is outputted (added to list and printed)
    resulting_sentences = []
    sentence_queue = [[starting_symbol]]
    while len(sentence_queue) > 0:
        sentence = sentence_queue.pop(0)
        if _all_leaves(sentence, grammar):
            # do final processing
            #   join sentence into one string
            #   fix apostrophes
            #   fix '*' ungrammatical symbols
            #   add a period
            #   fix double spaces from optionals
            sentence = (" ".join(sentence).replace(
                " '", "'").replace("* ", "*").replace("  ", " ") + ".")
            resulting_sentences.append(sentence)
            # print(sentence)
        else:
            sentence_queue.extend(
                _expand_first_non_terminal(sentence, grammar))

    return (starting_symbol, resulting_sentences)


def build_csv(grammar, starts, file_name):
    start_time = time.time()

    # type, sentence, grammaticality judgement (Y, N)
    results = [("Type", "Sentence", "Grammatical")]
    for start in starts:
        type, sentences = generate_sentences(grammar, start)
        # for now, grammaticality judgement is just based on
        # whether sentence starts with *.
        results.extend([(type, sentence, "N" if sentence[0] ==
                       "*" else "Y") for sentence in sentences])
    # build it
    try:
        with open(file_name, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(results)

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"{len(results)} line CSV file successfully saved to {file_name} in {elapsed_time:.4f} seconds.")

        return len(results)

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while building the CSV {str(e)}")

        return 0
