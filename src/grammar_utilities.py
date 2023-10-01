import time
import csv


def _expand_first_non_terminal(input_sentence: list, grammar: dict[str]) -> list[list]:
    """Returns all possible expansions of the provided sentence. For example, if an input is `["VP", "DP"]` and a 
    rule `VP -> V {NP | DP}` (formatted as json internally) the output would be `[["VP", "NP"], ["VP", "DP"]]`.

    Args:
        input_sentence (list): sentence which will have its first non-terminal entry expanded
        grammar (dict[str]): grammar to define what exactly is a non-terminal and what it should be replaced with

    Returns:
        list[list]: list of sentences, each one replaces the non-terminal with a possible output.
    """
    # element which will be expanded, either string non-terminal or array
    for element in input_sentence:
        if isinstance(element, list):
            first_non_terminal = element
            break
        elif element in grammar.keys():
            first_non_terminal = element
            break

    # non-terminals can be represented as strings which are also keys in the grammar,
    # or as arrays which define possible replacements without a new rule.
    if isinstance(element, str):
        return _expand_string_element(input_sentence, first_non_terminal, grammar.get(first_non_terminal))

    # grammar is not required since all replacement info is within arrays themselves
    elif isinstance(element, list):
        return _expand_array_element(input_sentence, first_non_terminal)


def _expand_string_element(input_sentence: list, replacement_non_terminal: str, replacement_expansions: list) -> list[list]:
    """There are two ways a non-terminal can be represented in an input grammar. This method handles string non-terminals.

    Args:
        input_sentence (list): sentence which will have its first non-terminal (string) entry expanded
        replacement_non_terminal (str): element which is to be expanded
        replacement_expansions (list): possible expansions (found within grammar) that will replace the non-terminal in the outputs

    Returns:
        list[list]: list of sentences. Original sentence with the non-terminal replaced with an expansion, for each expansion.
    """
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
    """There are two ways a non-terminal can be represented in an input grammar. This method handles list non-terminals.

    Args:
        input_sentence (list): sentence which will have its first non-terminal (list) entry expanded
        replacement_expansions (list): where to replace and what to replace it with. The list will be replaced with 
        each entry of the list to build the output

    Returns:
        list[list]: list of sentences. Original sentence with the non-terminal replaced with an expansion, for each expansion.
    """
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
    """Determines whether provided sentence, given grammar, is all non-terminals. If so, it cannot be expanded any further and is a complete output.

    Args:
        input_sentence (list): Sentence to be checked for non-terminals
        grammar (dict[str]): Defines grammar rules

    Returns:
        bool: Is the provided sentence, given grammar, all leaves? (No non-terminals)
    """
    for element in input_sentence:
        # OR--must be broken up
        if isinstance(element, list):
            return False
        # non-terminal
        if element in grammar.keys():
            return False
    # all elements must be leaves
    return True


def generate_sentences(grammar: dict[str], starting_symbol: str) -> tuple[str, list[str]]:
    """Generates ALL possible sentences from the grammar.

    Algorithm: 
    - queue starts with base node
    - until queue is empty, remove first sentence
    - this sentence's first non-terminal node is expanded and sentence is added to the queue
    - if sentence has no non-terminal nodes it is outputted (added to list and printed)

    Args:
        grammar (dict[str])
        starting_symbols (str): where in the grammar to start generating sentences. 

    Returns:
        tuple[str, list[str]]: First element is the type of sentence (just the starting symbol again).
        Second element is the lsit of all generated sentences.
    """
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


def build_csv(grammar, starts, file_name) -> int:
    """Generates sentences for each start and saves them to a csv file.

    Args:
        grammar (_type_): 
        starts (_type_): 
        file_name (_type_): 

    Raises:
        RuntimeError: Failed to write CSV

    Returns:
        int: number of total sentences generated  
    """
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
        print(f"{len(results) - 1} line CSV file successfully saved to {file_name} in {elapsed_time:.4f} seconds.")

        return len(results) - 1

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while building the CSV {str(e)}")

        return 0
