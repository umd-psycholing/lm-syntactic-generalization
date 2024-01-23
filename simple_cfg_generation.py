from nltk.grammar import CFG
import time


import generate_corpora as gc
import grammars
import surprisal

model = "grnn"

# cleft
cleft_grammar_c = CFG.fromstring(grammars.CLEFT_GRAMMAR_C)
output = gc.generate_all_sentence_tuples_from_grammar(cleft_grammar_c)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/cleft_grammar_c.json")
print("cleft c done")

cleft_grammar_i = CFG.fromstring(grammars.CLEFT_GRAMMAR_I)
output = gc.generate_all_sentence_tuples_from_grammar(cleft_grammar_i)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/cleft_grammar_i.json")
print(f"cleft i done")


# topic w/ intro
intro_topic_grammar_c = CFG.fromstring(grammars.INTRO_TOPIC_GRAMMAR_C)
output = gc.generate_all_sentence_tuples_from_grammar(intro_topic_grammar_c)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/intro_topic_grammar_c.json")
print("intro topic c done")

intro_topic_grammar_i = CFG.fromstring(grammars.INTRO_TOPIC_GRAMMAR_I)
output = gc.generate_all_sentence_tuples_from_grammar(intro_topic_grammar_i)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/intro_topic_grammar_i.json")
print("intro topic i done")


# topic NO intro
nointro_topic_grammar_c = CFG.fromstring(grammars.NOINTRO_TOPIC_GRAMMAR_C)
output = gc.generate_all_sentence_tuples_from_grammar(nointro_topic_grammar_c)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/nointro_topic_grammar_c.json")
print("no intro topic c done")

nointro_topic_grammar_i = CFG.fromstring(grammars.NOINTRO_TOPIC_GRAMMAR_I)
output = gc.generate_all_sentence_tuples_from_grammar(nointro_topic_grammar_i)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/nointro_topic_grammar_i.json")
print("no intro topic i done")

# tough movement
tough_grammar_c = CFG.fromstring(grammars.TOUGH_GRAMMAR_C)
output = gc.generate_all_sentence_tuples_from_grammar(tough_grammar_c)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/tough_grammar_c.json")
print("tough c done")

tough_grammar_i = CFG.fromstring(grammars.TOUGH_GRAMMAR_I)
output = gc.generate_all_sentence_tuples_from_grammar(tough_grammar_i)
for tup in output:
    surprisal.surprisal_effect_full_tuple(tup, model, True)
gc.corpus_to_json(
    output, f"grammar_outputs/simple_cfgs/{model}/tough_grammar_i.json")
print("tough i done")

print(f"CFGs and surprisals CFGs generated for {model}.")
