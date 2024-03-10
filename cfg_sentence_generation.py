# THIS FILE NEEDS TO BE MADE AUTOMATIC! #

from nltk.grammar import CFG


import generate_corpora as gc
import grammars
import surprisal

model = "gpt2"


def generate_and_calculate(cfg, filepath):
    output = gc.generate_all_sentence_tuples_from_grammar(cfg)
    for tup in output:
        surprisal.surprisal_effect_full_tuple(tup, model, True)
    gc.corpus_to_json(output, filepath)

    print(f"saved to {filepath}.")


# standard
# clefting
generate_and_calculate(CFG.fromstring(grammars.CLEFT_GRAMMAR_C),
                       f"grammar_outputs/experiment1/{model}/cleft_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.CLEFT_GRAMMAR_I),
                       f"grammar_outputs/experiment1/{model}/cleft_grammar_i.json")

# topicalization w/ intro
generate_and_calculate(CFG.fromstring(grammars.INTRO_TOPIC_GRAMMAR_C),
                       f"grammar_outputs/experiment1/{model}/intro_topic_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.INTRO_TOPIC_GRAMMAR_I),
                       f"grammar_outputs/experiment1/{model}/intro_topic_grammar_i.json")

# topicalization w/out intro
generate_and_calculate(CFG.fromstring(grammars.NOINTRO_TOPIC_GRAMMAR_C),
                       f"grammar_outputs/experiment1/{model}/nointro_topic_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.NOINTRO_TOPIC_GRAMMAR_I),
                       f"grammar_outputs/experiment1/{model}/nointro_topic_grammar_i.json")

# tough movement
generate_and_calculate(CFG.fromstring(grammars.TOUGH_GRAMMAR_C),
                       f"grammar_outputs/experiment1/{model}/tough_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.TOUGH_GRAMMAR_I),
                       f"grammar_outputs/experiment1/{model}/tough_grammar_i.json")

# training
# clefting
generate_and_calculate(CFG.fromstring(grammars.TRAINING_CLEFT_GRAMMAR_C),
                       f"grammar_outputs/training_sentences/{model}/cleft_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.TRAINING_CLEFT_GRAMMAR_I),
                       f"grammar_outputs/training_sentences/{model}/cleft_grammar_i.json")

# topicalization w/ intro
generate_and_calculate(CFG.fromstring(grammars.TRAINING_INTRO_TOPIC_GRAMMAR_C),
                       f"grammar_outputs/training_sentences/{model}/intro_topic_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.TRAINING_INTRO_TOPIC_GRAMMAR_I),
                       f"grammar_outputs/training_sentences/{model}/intro_topic_grammar_i.json")

# topicalization w/out intro
generate_and_calculate(CFG.fromstring(grammars.TRAINING_NOINTRO_TOPIC_GRAMMAR_C),
                       f"grammar_outputs/training_sentences/{model}/nointro_topic_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.TRAINING_NOINTRO_TOPIC_GRAMMAR_I),
                       f"grammar_outputs/training_sentences/{model}/nointro_topic_grammar_i.json")

# tough movement
generate_and_calculate(CFG.fromstring(grammars.TRAINING_TOUGH_GRAMMAR_C),
                       f"grammar_outputs/training_sentences/{model}/tough_grammar_c.json")
generate_and_calculate(CFG.fromstring(grammars.TRAINING_TOUGH_GRAMMAR_I),
                       f"grammar_outputs/training_sentences/{model}/tough_grammar_i.json")

"""
data = gc.corpus_from_json(
    "grammar_outputs/training_sentences/gpt2/cleft_grammar_c.json", True)
gc.simple_convert_to_csv(
    data, "grammar_outputs/training_sentences/csv/cleft_grammar_c.csv", "cleft", "gap")
"""
