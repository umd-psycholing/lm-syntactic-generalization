import random
import math
import collections
import itertools
import re

from nltk.parse import generate, earleychart
from nltk.grammar import CFG, Production
from nltk import Tree, Nonterminal

# these were orignally defined in this module, but were moved for clarity
# imported here so that any dependencies still have access
from sentence_tuples import TokenData, SentenceData, TupleSentenceData
from sentence_tuples import corpus_to_json, corpus_from_json, simple_convert_to_csv

import grammars

import itertools


def _get_terminal_productions_of_grammar_output(output_tokens: list[str], grammar: CFG) -> dict[Nonterminal, Production]:
    parser = earleychart.EarleyChartParser(grammar)
    trees = tuple(parser.parse(output_tokens))

    assert len(trees) > 0

    # recursive function unpacks parsed tree
    def _get_terminals(parser_tree: Tree):
        curr_terminals = []
        for daughter in parser_tree:
            if isinstance(daughter, Tree):
                curr_terminals += _get_terminals(daughter)
            else:
                curr_terminals += [(parser_tree.label(), daughter)]

        return curr_terminals

    terminals = _get_terminals(trees[0])
    nonterminal_to_production = {}
    # filter out guaranteed terminals (genitive, gap marker, etc.)
    for nonterminal, terminal in terminals:
        nonterminal = Nonterminal(nonterminal)
        if len(grammar._lhs_index[nonterminal]) > 1:
            nonterminal_to_production[nonterminal] = Production(
                nonterminal, [terminal])

    return nonterminal_to_production


def _find_accessible_lexical_productions_from_grammar(grammar):
    reachable = set()
    stack = [grammar._start]

    while stack:
        current_symbol = stack.pop()
        if current_symbol in reachable:
            continue
        reachable.add(current_symbol)

        for production in grammar.productions():
            if production.lhs() == current_symbol:
                for symbol in production.rhs():
                    if isinstance(symbol, Nonterminal):
                        stack.append(symbol)

    return reachable


def _process_tokens(output_tokens: list[str]) -> tuple[list[TokenData], TokenData]:
    tokens = []
    critical_tokens = []

    # flag
    between_underscores = False

    for token in output_tokens:
        if token == grammars.GAP_MARKER:
            if between_underscores:
                # if we were between underscores, we found the end--reset
                between_underscores = False
            else:
                # if we weren't, we found the start--set flag to True
                between_underscores = True
        else:
            # new_token = TokenData(text=str(token))
            tokens.append(token)
            if between_underscores:
                critical_tokens.append(token)

    return tokens, critical_tokens


def _tokenize(grammar_output) -> tuple[str, ...]:
    grammar_output = re.sub(r"(\w)\?", r"\1 ?", grammar_output)
    return tuple(grammar_output.split(" "))


def _fuse_apostrophes_in_tuple(input_tuple):
    result = []
    i = 0
    while i < len(input_tuple):
        current = input_tuple[i]
        if i + 1 < len(input_tuple) and input_tuple[i + 1] == "'s":
            result.append(current + input_tuple[i + 1])
            i += 2
        else:
            result.append(current)
            i += 1
    return tuple(result)


def _grammar_output_to_sentence(output_tokens: list[str]) -> SentenceData:
    # break into tokens
    broken_tokens = ()
    for t in output_tokens:
        broken_tokens += _tokenize(t)

    # merge possessives (gpt2 unmerges them automatically from merged, and we un-merge them before passing to GRNN)
    # maybe possessives should start un-merged, and then merged before passed to GPT2
    broken_tokens = _fuse_apostrophes_in_tuple(broken_tokens)

    # tokens to sentence

    grammatical = True
    # remove ungrammatical symbol (if necessary)
    if broken_tokens[0] == grammars.UNGRAMMATICAL:
        broken_tokens = broken_tokens[1:]
        grammatical = False

    # determines whether each token is critical, gets rid of GAP_MARKER
    processed_tokens, critical_tokens = _process_tokens(broken_tokens)

    return SentenceData(
        original_tokens=output_tokens,
        grammatical=grammatical,
        processed_tokens=processed_tokens,
        critical_tokens=critical_tokens
    )


def generate_all_sentences_from_grammar(grammar: CFG) -> tuple[SentenceData]:
    output = []

    # generate +filler,+gap (grammatical) FG
    grammar._start = grammars.AB_CONDITION

    filler_gap_tokens = list(generate.generate(grammar))
    filler_gap_sentences = [
        _grammar_output_to_sentence(tokens) for tokens in filler_gap_tokens]
    output.extend(filler_gap_sentences)

    # generate -filler,+gap (grammatical) XG
    grammar._start = grammars.XB_CONDITION

    no_filler_gap_tokens = list(generate.generate(grammar))
    no_filler_gap_sentences = [
        _grammar_output_to_sentence(tokens) for tokens in no_filler_gap_tokens]
    output.extend(no_filler_gap_sentences)

    # generate +filler,-gap (grammatical) FX
    grammar._start = grammars.AX_CONDITION

    filler_no_gap_tokens = list(generate.generate(grammar))
    filler_no_gap_sentences = [
        _grammar_output_to_sentence(tokens) for tokens in filler_no_gap_tokens]
    output.extend(filler_no_gap_sentences)

    # generate -filler,-gap (grammatical) XX
    grammar._start = grammars.XX_CONDITION

    no_filler_no_gap_tokens = list(generate.generate(grammar))
    no_filler_no_gap_sentences = [
        _grammar_output_to_sentence(tokens) for tokens in no_filler_no_gap_tokens]
    output.extend(no_filler_no_gap_sentences)

    return output


def generate_all_sentence_tuples_from_grammar(grammar: CFG) -> tuple[TupleSentenceData]:
    # find all lex. productions. Using XX and AB should lead to all lexical productions

    grammar._start = grammars.XX_CONDITION
    lexical_productions = _find_accessible_lexical_productions_from_grammar(
        grammar)

    grammar._start = grammars.AB_CONDITION
    # add potential other lexical items
    lexical_productions = lexical_productions.union(
        _find_accessible_lexical_productions_from_grammar(grammar))

    # it is possible that a production is used in XX and is not in AB vice versa.
    # this case *will* lead to additional tuples, but every case should be found.

    # find types which are not constant
    lexical_types = {key: value for key,
                     value in grammar._lhs_index.items() if len(value) > 1 and key in lexical_productions}

    # productions that shouldn't (left hand side is a non-lexical type)
    constant_productions = list(
        filter(lambda production: production.lhs() not in lexical_types.keys(),
               grammar._productions))

    possible_used_productions = list(
        itertools.product(*lexical_types.values()))

    # order: +A,+B | -A,+B | +A,-B | -A,-B
    tuples = []
    # now, for each possible_lexical_selection, build the grammar & generate each tree
    for used_production_list in possible_used_productions:
        # build new productions
        new_productions = constant_productions + list(used_production_list)
        # build new grammar
        new_grammar = CFG(productions=new_productions,
                          start=grammars.XX_CONDITION)

        # generate new sentence for each type (only one possible, n=1 is there for clarity)
        tuples.append(
            TupleSentenceData(
                s_ab=_grammar_output_to_sentence(
                    list(generate.generate(new_grammar, grammars.AB_CONDITION, n=1))[0]),
                s_xb=_grammar_output_to_sentence(
                    list(generate.generate(new_grammar, grammars.XB_CONDITION, n=1))[0]),
                s_ax=_grammar_output_to_sentence(
                    list(generate.generate(new_grammar, grammars.AX_CONDITION, n=1))[0]),
                s_xx=_grammar_output_to_sentence(
                    list(generate.generate(new_grammar, grammars.XX_CONDITION, n=1))[0]),
            )
        )

    return tuples


# NOTE: The following two functions implement an outdated method of corpus generation.
# Modified from Lan, et al. (2023)
# Function get_terminal_productions is slightly different in our implementation,
# accounting for the variation in code seen where that function is invoked.
def generate_train_test_grammatical_sentence_sets_from_grammar(
    grammar: CFG, split_ratio: float = 0.65,
    random_seed: int = grammars.RANDOM_SEED,
    base_grammatical_condition: Nonterminal = grammars.AB_CONDITION) -> tuple[tuple[SentenceData],
                                                                              tuple[SentenceData]]:

    # start from base grammatical condition, which for filler gap constructions is +filler, +gap.
    # for island constructions it is +extractible (no island), +gap
    grammar._start = base_grammatical_condition

    all_sentences = []

    # (NAME1, 'John') -> set(1, 4, 6, ...)
    single_lex_choice_to_sentence_idxs = collections.defaultdict(set)

    # 'NAME1' -> set('John', 'Michael', 'Harold', ...)
    all_lex_choices_per_category = collections.defaultdict(set)

    # set of all lexical choices (tuples) -> x (index)
    full_lex_choices_to_sentence_idx = {}

    # for each generated sentence and its index
    for i, sentence_tokens in enumerate(generate.generate(grammar)):
        # find its productions (which lexical items did it use)
        lex_choices = frozenset(((key, value) for key, value in
                                 _get_terminal_productions_of_grammar_output(sentence_tokens, grammar).items()))
        # (this sentence's used lexical items) -> this sentence's index
        full_lex_choices_to_sentence_idx[lex_choices] = i

        # for each category and lexical choice this sentence made
        for category, lex_choice in lex_choices:
            # assemble set of lexical choices per category (for use later when reserving test lexical items)
            all_lex_choices_per_category[category].add(lex_choice)
            # (category, lexical choice) -> all s_xx sentences which chose that lexical item for that category
            single_lex_choice_to_sentence_idxs[(category, lex_choice)].add(i)

        # add SentenceData form of sentence
        all_sentences.append((_grammar_output_to_sentence(sentence_tokens)))

    # category -> lexical choices (set aside for testing set sentences)
    test_lex_choices = {}

    random.seed(random_seed)  # set random seed

    # for each category and its lexical choices
    for category, lex_choices in all_lex_choices_per_category.items():
        # reserve (1 - split_ratio) proportion of lexical choices for testing set

        # NOTE: Lan, et al. (2022) claims to reserve 65% of lexical choices per category for test sentences.
        # In fact, they reserve 35% (rounded up) for test lexical choices.
        # Then, of all sentences generated, all sentences which include two lexical choices reserved for
        # test sentences are removed in order to generate the training set.
        #   Ends up removing all of the testing set items by default. (All lexical choices are test-reserved)

        num_test_lex_choices = math.ceil((1 - split_ratio) * len(lex_choices))
        # select that many randomly from the category's lexical choices
        category_test_lex_choices = tuple(
            random.sample(sorted(lex_choices), num_test_lex_choices)
        )
        test_lex_choices[category] = category_test_lex_choices

    all_full_test_lex_choices = tuple(
        map(
            frozenset,
            itertools.product(
                # find all combinations of test lexical choices which can then be used
                # alongside full_lex_choices_to_sentence_idx to retrieve the sentences
                # generated from those lexical choices.
                *[
                    [(cat, lex_choice) for lex_choice in test_lex_choices[cat]]
                    for cat in test_lex_choices
                ]
            ),
        )
    )

    # get indices of sentences using test-reserved lexical choices
    test_idxs = set()
    # for each test full lexical choice (full meaning for all categories)
    for full_test_lex_choice in all_full_test_lex_choices:
        # add sentence constructed using that set of choices
        test_idxs.add(full_lex_choices_to_sentence_idx[full_test_lex_choice])
    # print(test_lex_choices)  # debug
    # construct pairs of test lexical choices (to be removed from training)
    test_lex_choice_pairs = []
    # for each pair of categories
    for (cat1, cat1_choices), (cat2, cat2_choices) in itertools.combinations(
        test_lex_choices.items(), 2
    ):
        # for each pair of lexical choices within the selected categories
        for lex_choice1, lex_choice2 in itertools.product(cat1_choices, cat2_choices):
            # append that (category, lexical choice), (category, lexical choice) pairing
            test_lex_choice_pairs.append(
                ((cat1, lex_choice1), (cat2, lex_choice2)))

    # training sentences starts with EVERY sentence
    training_idxs = set(range(len(all_sentences)))

    # for each pair of (category, lexical choice), (category, lexical choice)
    for (cat1, lex_choice1), (cat2, lex_choice2) in test_lex_choice_pairs:
        # get the indices of all sentences which use each category
        sentences1 = single_lex_choice_to_sentence_idxs[(cat1, lex_choice1)]
        sentences2 = single_lex_choice_to_sentence_idxs[(cat2, lex_choice2)]
        # subtract sentences which overlap (use both cat1's choice1 and cat2's choice2)
        training_idxs -= (sentences1 & sentences2)

    training_sentences = tuple(all_sentences[i] for i in sorted(training_idxs))
    test_sentences = tuple(all_sentences[i] for i in sorted(test_idxs))

    return training_sentences, test_sentences


# mimics above function, but returns corpus using TupleSentenceData format
def generate_train_test_tuples_from_grammar(grammar: CFG, split_ratio: float = 0.65):
    # really simple method here

    # first generate all the training, test s_fg sentences, ...
    training, testing = generate_train_test_grammatical_sentence_sets_from_grammar(grammar=grammar,
                                                                                   split_ratio=split_ratio)

    # ... then find all the tuples which contain those sentences.
    all_tuples = generate_all_sentence_tuples_from_grammar(grammar=grammar)
    training_tuples = [
        sentence_tuple for sentence_tuple in all_tuples if sentence_tuple.s_ab in training]
    testing_tuples = [
        sentence_tuple for sentence_tuple in all_tuples if sentence_tuple.s_ab in testing]

    return training_tuples, testing_tuples
###
