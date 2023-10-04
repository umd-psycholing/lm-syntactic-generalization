import time
import csv
import re


class SentenceInfo:
    def __init__(self, list_representation):
        self.list_representation = list_representation

        # ungrammatical if starts with *
        self.grammatical = False if list_representation[0] == '*' else True
        # get rid of '#' joining character and '*' ungrammatical character
        self.text = re.sub(r'(\s*\*\s*)|(\s*#\s*)', '',
                           " ".join(list_representation))

        self.region_start = self.text.find('/')

        # account for removal of '/'s and automatically inserted spaces.
        self.region_end = self.text.rfind('/') - 3
        if self.region_end < 0 or self.region_start == self.region_end:
            self.region_end = -1

        # remove ' / '.
        self.text = (re.sub(r'\s*\/\s*', ' ', self.text)).strip()

    
    def is_training(self, free_lexical_items: list[str]=None):
        if not free_lexical_items:
            return None
        
        # determine whether it is a training_set item
        for lexical_item in free_lexical_items:
            if lexical_item[0] in self.list_representation:
                return True
        return False


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
        if isinstance(element, list) or element in grammar.keys():
            first_non_terminal = element

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


def generate_sentences(grammar: dict[str], starting_symbol: str) -> tuple[str, list[SentenceInfo]]:
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
        tuple[str, list[SentenceInfo]]: First element is the type of sentence (just the starting symbol again).
        Second element is the list of all generated sentences.
    """
    resulting_sentences = []
    sentence_queue = [[starting_symbol]]
    while len(sentence_queue) > 0:
        sentence = sentence_queue.pop(0)
        if _all_leaves(sentence, grammar):
            # join sentence into one string
            # '#' represents no character, meaning any automatically generated spaces should be removed.
            #   as a side-effect, that makes '#' perfect for representing optional symbols.
            #   ex: [["optional", "#"]] will have sentences with "optional" and some with it removed.
            #       this is better than [["optional"], [" "]] because there will not be double-spaces generated by " ".join(...).
            # fix space after '*' ungrammatical symbols
            # add a period to the end of the sentence

            """
            sentence = re.sub(r'\s*\*\s*', '*',
                              re.sub(r'\s*#\s*', '', " ".join(sentence) + "."))

            resulting_sentences.append(sentence)
            """

            resulting_sentences.append(SentenceInfo(sentence))
            # print(sentence)
        else:
            sentence_queue.extend(
                _expand_first_non_terminal(sentence, grammar))

    return (starting_symbol, resulting_sentences)


def build_csv(grammar, starts, file_name, reserved_types=None) -> int:
    """Generates sentences for each start and saves them to a csv file.

    Args:
        grammar (_type_): 
        starts (_type_): 
        file_name (_type_): 
        reserved_types (_type_):

    Raises:
        RuntimeError: Failed to write CSV

    Returns:
        int: number of total sentences generated  
    """
    start_time = time.time()

    # type, sentence, grammaticality judgement (Y, N), region start, region end, is training 
    results = [("Type", "Sentence", "Grammatical", 
                "Region Start", "Region End", "Critical String", 
                "Is Training")]
    
    # unpack reserved_types
    reserved_lexical_items = None
    if reserved_types:
        reserved_lexical_items = []
        for reserved_type in reserved_types:
            non_terminals = grammar[reserved_type]
            reserved_lexical_items.extend(non_terminals[:int(len(non_terminals) * .35)])
        print(reserved_lexical_items)

    for start in starts:
        type, sentences = generate_sentences(grammar, start)
        for sentence in sentences:
            is_training = sentence.is_training(reserved_lexical_items)

            results.append([type, sentence.text, sentence.grammatical,
                           sentence.region_start, sentence.region_end, sentence.text[sentence.region_start:sentence.region_end], is_training])
    
    # build it
    try:
        with open(file_name, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(results)

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(
            f"{len(results) - 1} line CSV file saved to {file_name} in {elapsed_time:.4f} seconds.")

        return len(results) - 1

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while building the CSV {str(e)}")

        return 0
