from nltk.grammar import CFG

import generate_corpora as gc
import grammars

# cleft
cleft_grammar_c = CFG.fromstring(grammars.CLEFT_GRAMMAR_C)
output = gc.generate_all_sentence_tuples_from_grammar(cleft_grammar_c)
gc.corpus_to_json(output, "grammar_outputs/simple_cfgs/cleft_grammar_c.json")

cleft_grammar_i = CFG.fromstring(grammars.CLEFT_GRAMMAR_I)
output = gc.generate_all_sentence_tuples_from_grammar(cleft_grammar_i)
gc.corpus_to_json(output, "grammar_outputs/simple_cfgs/cleft_grammar_i.json")


# topic
topic_grammar_c = CFG.fromstring(grammars.TOPIC_GRAMMAR_C)
output = gc.generate_all_sentence_tuples_from_grammar(topic_grammar_c)
gc.corpus_to_json(output, "grammar_outputs/simple_cfgs/topic_grammar_c.json")

topic_grammar_i = CFG.fromstring(grammars.TOPIC_GRAMMAR_I)
output = gc.generate_all_sentence_tuples_from_grammar(topic_grammar_i)
gc.corpus_to_json(output, "grammar_outputs/simple_cfgs/topic_grammar_i.json")


# tough movement
tough_grammar_c = CFG.fromstring(grammars.TOUGH_GRAMMAR_C)
output = gc.generate_all_sentence_tuples_from_grammar(tough_grammar_c)
gc.corpus_to_json(output, "grammar_outputs/simple_cfgs/tough_grammar_c.json")

tough_grammar_i = CFG.fromstring(grammars.TOUGH_GRAMMAR_I)
output = gc.generate_all_sentence_tuples_from_grammar(tough_grammar_i)
gc.corpus_to_json(output, "grammar_outputs/simple_cfgs/tough_grammar_i.json")

print("simple CFGs generated.")
