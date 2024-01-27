from nltk import Nonterminal

RANDOM_SEED = 118755212

# markers
UNGRAMMATICAL = "*"
GAP_MARKER = "_"
TUPLE_DIVIDER = "&"

# start constants
ALL_CONDITION = Nonterminal("S")
AB_CONDITION = Nonterminal("S_AB")
XB_CONDITION = Nonterminal("S_XB")
AX_CONDITION = Nonterminal("S_AX")
XX_CONDITION = Nonterminal("S_XX")

# all of this is shared between all grammars. (might not all be used, but its all available)
# written here to reduce duplicative code
SHARED_GRAMMAR = """
    S -> S_AB | S_XB | S_AX | S_XX
    GAP_MARKER -> "_"
    UNGRAMMATICAL -> "*"
    GEN -> "'s" 
    NAME1 -> "Michael" | "Ashley" | "Daniel" | "John" | "Brandon" | "William" | "Nicole" | "Eric" | "Melissa" | "Timothy"
    NAME2 -> "Christopher" | "Jennifer" | "David"
    NAME3 -> "Jessica" | "Joshua" | "James"
    NAME4 -> "Matthew" | "you"
"""

# Parasitic Gap constructions
PG_GRAMMARS = [
    SHARED_GRAMMAR +  # PG 1
    """
    S_AB -> PREAMBLE F G
    S_XB -> UNGRAMMATICAL PREAMBLE XF G
    S_AX -> UNGRAMMATICAL PREAMBLE F XG
    S_XX -> PREAMBLE XF XG
    PREAMBLE -> "I know"
    F -> "who" NAME1 GEN ADJ SUBJ V
    XF -> "that" NAME1 GEN ADJ SUBJ NAME2 V
    G -> GAP_MARKER ADJUNCT GAP_MARKER
    XG -> GAP_MARKER NAME4 GAP_MARKER ADJUNCT
    SUBJ -> "talking to" | "attitude towards" | "friendship with" | "praising of"
    ADJ -> "recent" | "current" 
    V -> "bothered" | "distracted" | "worried" | "annoyed" 
    ADJUNCT -> "recently" | "yesterday" | "lately"
    """,
    SHARED_GRAMMAR +  # PG 2
    """
    PREAMBLE -> "I know"
    S_AB -> PREAMBLE F G
    S_XB -> UNGRAMMATICAL PREAMBLE XF G
    S_AX -> UNGRAMMATICAL PREAMBLE F XG
    S_XX -> PREAMBLE XF XG
    F -> "what" SUBJ V1 ADV V2
    XF -> "that" SUBJ V1 OBJ1 ADV V2
    G -> GAP_MARKER ADJUNCT GAP_MARKER
    XG -> GAP_MARKER OBJ2 GAP_MARKER ADJUNCT
    SUBJ -> "the attempt to"
    V1 -> "repair" | "fix" | "overhaul" | "rebuild"
    V2 -> "damaged" | "destroyed" | "ruined" | "wrecked"
    OBJ1 -> "the car" | "the bike" | "the washing machine" | "the drier" | "the ceiling" | "the apartment"
    OBJ2 -> "it"
    ADV -> "eventually" | "finally"
    ADJUNCT -> "nevertheless" | "nonetheless"
    """,
    SHARED_GRAMMAR +  # PG 3
    """
    PREAMBLE -> "I know"
    S_AB -> PREAMBLE F G
    S_XB -> UNGRAMMATICAL PREAMBLE XF G
    S_AX -> UNGRAMMATICAL PREAMBLE F XG
    S_XX -> PREAMBLE XF XG
    F -> "who" SUBJ NAME1 ADV V1
    XF -> "that" SUBJ NAME1 ADV V1 NAME3
    G -> V2 GAP_MARKER ADJUNCT GAP_MARKER
    XG -> V2 GAP_MARKER NAME4 GAP_MARKER ADJUNCT
    SUBJ -> "the" SUBJ1 "that"
    SUBJ1 -> "fact" | "idea" | "rumor"
    ADV -> "secretly" | "really" | "absolutely" | "actually"
    V1 -> "liked" | "loved" | "hated" | "fancied"
    V2 -> "surprised" | "shocked" | "irritated"
    ADJUNCT -> "today" | "yesterday" | "recently"
    """,
    SHARED_GRAMMAR +  # PG 4
    """
    PREAMBLE -> "I know"
    S_AB -> PREAMBLE F G
    S_XB -> UNGRAMMATICAL PREAMBLE XF G
    S_AX -> UNGRAMMATICAL PREAMBLE F XG
    S_XX -> PREAMBLE XF XG
    F -> "what the" SUBJ "to" V1
    XF -> "that the" SUBJ "to" V1 OBJ1
    G -> V3 V2 GAP_MARKER ADJUNCT GAP_MARKER
    XG -> V3 V2 GAP_MARKER OBJ2 GAP_MARKER ADJUNCT
    SUBJ -> "political campaign" | "recommendation" | "legislation" | "suggestion"
    V1 -> "preserve" | "help" | "save"
    OBJ1 -> "nature" | "the environment" | "the rain forests" | "biodiversity"
    V3 -> "made people" | "caused people to"
    V2 -> "harm" | "hurt"
    OBJ2 -> "animals" | "wildlife" | "plants" | "trees"
    ADJUNCT -> "nevertheless" | "nonetheless"
    """
]

# Across-the-Board constructions
ATB_GRAMMARS = [
    SHARED_GRAMMAR +  # ATB 1
    """
    S_AB -> PREAMBLE F G
    S_XB -> UNGRAMMATICAL PREAMBLE XF G
    S_AX -> UNGRAMMATICAL PREAMBLE F XG
    S_XX -> PREAMBLE XF XG
    PREAMBLE -> "I know"
    F -> "what" NAME1 V1
    XF -> "that" NAME1 V1 OBJ1
    CONN -> "yesterday and will"
    G -> CONN V2 GAP_MARKER ADJUNCT GAP_MARKER
    XG -> CONN V2 GAP_MARKER OBJ2 GAP_MARKER ADJUNCT
    V1 -> "looked for" | "searched everywhere for" | "found" | "bought" | "purchased" | "went shopping for"
    OBJ1 -> "food" | "bread" | "meat" | "cheese" | "candy"
    V2 -> "devour" | "serve" | "donate" | "distribute"
    OBJ2 -> "it" | "fish" | "snacks"
    ADJUNCT -> "tomorrow" | "soon" | "tonight" | "today" | "shortly" | "quickly"
    """,
    SHARED_GRAMMAR +  # ATB 2
    """
    S_AB -> PREAMBLE F G
    S_XB -> UNGRAMMATICAL PREAMBLE XF G
    S_AX -> UNGRAMMATICAL PREAMBLE F XG
    S_XX -> PREAMBLE XF XG
    PREAMBLE -> "I know"
    F -> "who" NAME1 V1
    XF -> "that" NAME1 V1 NAME2
    CONN -> "last year and"
    G -> CONN V2 GAP_MARKER ADJUNCT GAP_MARKER
    XG -> CONN V2 GAP_MARKER NAME4 GAP_MARKER ADJUNCT
    V1 -> "talked to" | "called" | "texted" | "yelled at" | "humiliated"
    V2 -> "argued with" | "had a fight with" | "made peace with" | "stopped talking to" | "fell in love with" | "started to like"
    ADJUNCT -> "today" | "recently" | "lately"
    """,
    SHARED_GRAMMAR +  # ATB 3
    """
    S_AB -> PREAMBLE F G
    S_XB -> UNGRAMMATICAL PREAMBLE XF G
    S_AX -> UNGRAMMATICAL PREAMBLE F XG
    S_XX -> PREAMBLE XF XG
    PREAMBLE -> "I know"
    F -> "who" NAME1 V1
    XF -> "that" NAME1 V1 NAME2
    CONN -> "and"
    G -> CONN V2 GAP_MARKER ADJUNCT GAP_MARKER
    XG -> CONN V2 GAP_MARKER NAME4 GAP_MARKER ADJUNCT
    V1 -> "saw" | "spotted" | "noticed" | "looked at"
    V2 -> "helped" | "played with" | "started to like" | "fell in love with"
    ADJUNCT -> "today" | "yesterday" | "recently" | "lately"
    """
]

# E = extractible, meaning no island.
#   This was done to ensure that S_XX (island, no gap) was represented by the lack of the two variables,
#   because generation of sentences depends on S_XX containing each lexical choice that can be made for a tuple
# G = gap position
"""
AB: It is these snacks that Mary knows Patricia bought yesterday.
XB: It is these snacks that Mary knows the reason Patricia bought yesterday.
AX: It is these snacks that Mary knows Patricia bought the cheese yesterday.
XX: It is these snacks that Mary knows the reason Patricia bought the cheese yesterday.
A = non-island, B = gap
X = island, X = no gap
"""
ISLAND_GRAMMAR = SHARED_GRAMMAR + """
    S_AB -> PREAMBLE E G
    S_XB -> UNGRAMMATICAL PREAMBLE XE G
    S_AX -> UNGRAMMATICAL PREAMBLE E XG
    S_XX -> UNGRAMMATICAL PREAMBLE XE XG
    PREAMBLE -> "It is" 
    E -> OBJ1 COMP N1 V1
    XE -> OBJ1 COMP N1 V1 IDENTIFIER
    G -> N2 V2 GAP_MARKER ADV GAP_MARKER
    XG -> N2 V2 GAP_MARKER OBJ2 GAP_MARKER ADV
    OBJ1 -> "these snacks" | "those boots" | "her books"
    COMP -> "that"
    N1 -> "Mary" | "Jennifer"
    N2 -> "Patricia" | "Linda"
    V1 -> "knows" | "heard" | "remembers" | "believes"
    IDENTIFIER -> "the reason" | "the claim" | "the rumor"
    V2 -> "bought" | "saw" | "forgot"
    OBJ2 -> "the cheese" | "your hat" | "her keys"
    ADV -> "yesterday" | "recently" | "earlier" 
"""


# experiment 1 CFGs start here
# cleft
CLEFT_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> PREAMBLE E G
	S_XB -> UNGRAMMATICAL PREAMBLE XE G
	S_AX -> UNGRAMMATICAL PREAMBLE E XG
	S_XX -> PREAMBLE XE XG
	PREAMBLE -> "It is"
	E -> OBJ1 COMP N1 V1
	XE -> CLEFTADJ COMP N1 V1
    G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	OBJ1 -> "these snacks" | "those boots" | "her books"
	CLEFTADJ -> "apparent" | "clear" | "disappointing"
	COMP -> "that"
	N1 -> "Mary" | "Jennifer"
	V1 -> "bought" | "saw" | "forgot"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	OBJ2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "yesterday" | "recently" | "earlier"
"""

CLEFT_GRAMMAR_I = SHARED_GRAMMAR + """
	S_AB -> UNGRAMMATICAL PREAMBLE E I G
	S_XB -> UNGRAMMATICAL PREAMBLE XE I G
	S_AX -> UNGRAMMATICAL PREAMBLE E I XG
	S_XX -> PREAMBLE XE I XG
	PREAMBLE -> "It is"
	E -> OBJ1 COMP N1 V1
	XE -> CLEFTADJ COMP N1 V1
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER 
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	OBJ1 -> "these snacks" | "those boots" | "her books"
	CLEFTADJ -> "apparent" | "clear" | "disappointing"
	COMP -> "that"
	N1 -> "Mary" | "Jennifer"
	V1 -> "bought" | "saw" | "forgot"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	OBJ2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "yesterday" | "recently" | "earlier"
"""

# topic w/ intro
INTRO_TOPIC_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> E G
	S_XB -> UNGRAMMATICAL XE G
	S_AX -> UNGRAMMATICAL E XG
	S_XX -> XE XG
	E -> TOPIC N1 V1
	XE -> INTRO N1 V1
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these snacks," | "those boots," | "her books,"
    INTRO -> "in fact," | "of course," | "clearly,"
    COMP -> "that"
	N1 -> "Mary" | "Jennifer"
	V1 -> "bought" | "saw" | "forgot"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	OBJ2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "yesterday" | "recently" | "earlier"
"""

INTRO_TOPIC_GRAMMAR_I = SHARED_GRAMMAR + """
    S_AB -> UNGRAMMATICAL E I G
	S_XB -> UNGRAMMATICAL XE I G
	S_AX -> UNGRAMMATICAL E I XG
	S_XX -> XE I XG
	E -> TOPIC N1 V1
	XE -> INTRO N1 V1
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these snacks," | "those boots," | "her books,"
    INTRO -> "in fact," | "of course," | "clearly,"
    COMP -> "that"
	N1 -> "Mary" | "Jennifer"
	V1 -> "bought" | "saw" | "forgot"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	OBJ2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "yesterday" | "recently" | "earlier"
"""

# topic NO intro
NOINTRO_TOPIC_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> E G
	S_XB -> UNGRAMMATICAL XE G
	S_AX -> UNGRAMMATICAL E XG
	S_XX -> XE XG
	E -> TOPIC N1 V1
	XE -> N1 V1
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these snacks," | "those boots," | "her books,"
    COMP -> "that"
	N1 -> "Mary" | "Jennifer"
	V1 -> "bought" | "saw" | "forgot"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	OBJ2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "yesterday" | "recently" | "earlier"
"""

NOINTRO_TOPIC_GRAMMAR_I = SHARED_GRAMMAR + """
    S_AB -> UNGRAMMATICAL E I G
	S_XB -> UNGRAMMATICAL XE I G
	S_AX -> UNGRAMMATICAL E I XG
	S_XX -> XE I XG
	E -> TOPIC N1 V1
	XE -> N1 V1
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these snacks," | "those boots," | "her books,"
    COMP -> "that"
	N1 -> "Mary" | "Jennifer"
	V1 -> "bought" | "saw" | "forgot"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	OBJ2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "yesterday" | "recently" | "earlier"
"""

# tough movement
TOUGH_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> E G
	S_XB -> UNGRAMMATICAL XE G
	S_AX -> UNGRAMMATICAL E XG
	S_XX -> XE XG
	E -> N1 TOUGH INF 
	XE -> IT TOUGH INF
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER N2 GAP_MARKER ADV
    COMP -> "that"
	N1 -> "this snack" | "the shoe" | "her book"
    IT -> "it"
    TOUGH -> "is impossible" | "is easy" | "is difficult"
	INF -> "to find" | "to see" | "to lose"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	N2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "with a friend" | "in the park" | "on Saturdays"
"""

TOUGH_GRAMMAR_I = SHARED_GRAMMAR + """
    S_AB -> UNGRAMMATICAL E I G
	S_XB -> UNGRAMMATICAL XE I G
	S_AX -> UNGRAMMATICAL E I XG
	S_XX -> XE I XG
	E -> N1 TOUGH INF 
	XE -> IT TOUGH INF
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER N2 GAP_MARKER ADV
    COMP -> "that"
	N1 -> "this snack" | "the shoe" | "her book"
    IT -> "it"
    TOUGH -> "is impossible" | "is easy" | "is difficult"
	INF -> "to find" | "to see" | "to lose"
    RC_OBJ -> "the bag" | "a car" | "the house"
    V2 -> "carried" | "held" | "contained"
	N2 -> "the cheese" | "your hat" | "her keys"
	ADV -> "with a friend" | "in the park" | "on Saturdays"
"""


# training set cfgs
# cleft
TRAINING_CLEFT_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> PREAMBLE E G
	S_XB -> UNGRAMMATICAL PREAMBLE XE G
	S_AX -> UNGRAMMATICAL PREAMBLE E XG
	S_XX -> PREAMBLE XE XG
	PREAMBLE -> "It is"
	E -> OBJ1 COMP N1 V1
	XE -> CLEFTADJ COMP N1 V1
    G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	OBJ1 -> "these cakes" | "those balls" | "her gloves"
	CLEFTADJ -> "obvious" | "surprising" | "annoying"
	COMP -> "that"
	N1 -> "Susan" | "Michelle"
	V1 -> "lifted" | "stole" | "threw"
    RC_OBJ -> "the box" | "this truck" | "her backpack"
    V2 -> "hauled" | "had" | "transported"
	OBJ2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""

TRAINING_CLEFT_GRAMMAR_I = SHARED_GRAMMAR + """
	S_AB -> UNGRAMMATICAL PREAMBLE E I G
	S_XB -> UNGRAMMATICAL PREAMBLE XE I G
	S_AX -> UNGRAMMATICAL PREAMBLE E I XG
	S_XX -> PREAMBLE XE I XG
	PREAMBLE -> "It is"
	E -> OBJ1 COMP N1 V1
	XE -> CLEFTADJ COMP N1 V1
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	OBJ1 -> "these cakes" | "those balls" | "her gloves"
	CLEFTADJ -> "obvious" | "surprising" | "annoying"
	COMP -> "that"
	N1 -> "Susan" | "Michelle"
	V1 -> "lifted" | "stole" | "threw"
    RC_OBJ -> "the box" | "this truck" | "her backpack"
    V2 -> "hauled" | "had" | "transported"
	OBJ2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""

# topic w/ intro
TRAINING_INTRO_TOPIC_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> E G
	S_XB -> UNGRAMMATICAL XE G
	S_AX -> UNGRAMMATICAL E XG
	S_XX -> XE XG
	E -> TOPIC N1 V1
	XE -> INTRO N1 V1
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these cakes," | "those balls," | "her gloves,"
    INTRO -> "indeed," | "meanwhile," | "still,"
    COMP -> "that"
	N1 -> "Susan" | "Michelle"
	V1 -> "lifted" | "stole" | "threw"
    RC_OBJ -> "the box" | "this truck" | "her backpack"
    V2 -> "hauled" | "had" | "transported"
	OBJ2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""

TRAINING_INTRO_TOPIC_GRAMMAR_I = SHARED_GRAMMAR + """
    S_AB -> UNGRAMMATICAL E I G
	S_XB -> UNGRAMMATICAL XE I G
	S_AX -> UNGRAMMATICAL E I XG
	S_XX -> XE I XG
	E -> TOPIC N1 V1
	XE -> INTRO N1 V1
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these cakes," | "those balls," | "her gloves,"
    INTRO -> "indeed," | "meanwhile," | "still,"
    COMP -> "that"
	N1 -> "Susan" | "Michelle"
	V1 -> "lifted" | "stole" | "threw"
    RC_OBJ -> "the box" | "this truck" | "her backpack"
    V2 -> "hauled" | "had" | "transported"
	OBJ2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""

# topic NO intro
TRAINING_NOINTRO_TOPIC_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> E G
	S_XB -> UNGRAMMATICAL XE G
	S_AX -> UNGRAMMATICAL E XG
	S_XX -> XE XG
	E -> TOPIC N1 V1
	XE -> N1 V1
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these cakes," | "those balls," | "her gloves,"
    COMP -> "that"
	N1 -> "Susan" | "Michelle"
	V1 -> "lifted" | "stole" | "threw"
    RC_OBJ -> "the box" | "this truck" | "her backpack"
    V2 -> "hauled" | "had" | "transported"
	OBJ2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""

TRAINING_NOINTRO_TOPIC_GRAMMAR_I = SHARED_GRAMMAR + """
    S_AB -> UNGRAMMATICAL E I G
	S_XB -> UNGRAMMATICAL XE I G
	S_AX -> UNGRAMMATICAL E I XG
	S_XX -> XE I XG
	E -> TOPIC N1 V1
	XE -> N1 V1
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER OBJ2 GAP_MARKER ADV
	TOPIC -> "these cakes," | "those balls," | "her gloves,"
    COMP -> "that"
	N1 -> "Susan" | "Michelle"
	V1 -> "lifted" | "stole" | "threw"
    RC_OBJ -> "the box" | "this truck" | "her backpack"
    V2 -> "hauled" | "had" | "transported"
	OBJ2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""

# tough movement
TRAINING_TOUGH_GRAMMAR_C = SHARED_GRAMMAR + """
	S_AB -> E G
	S_XB -> UNGRAMMATICAL XE G
	S_AX -> UNGRAMMATICAL E XG
	S_XX -> XE XG
	E -> N1 TOUGH INF 
	XE -> IT TOUGH INF
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER N2 GAP_MARKER ADV
    COMP -> "that"
	N1 -> "the pizza" | "that ball" | "his wallet"
    IT -> "it"
    TOUGH -> "is tough" | "is annoying" | "is simple" | "is embarrassing"
	INF -> "to steal" | "to throw" | "to lift"
    RC_OBJ -> "the box" | "this truck" | "his backpack"
    V2 -> "hauled" | "had" | "transported"
	N2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""

TRAINING_TOUGH_GRAMMAR_I = SHARED_GRAMMAR + """
    S_AB -> UNGRAMMATICAL E I G
	S_XB -> UNGRAMMATICAL XE I G
	S_AX -> UNGRAMMATICAL E I XG
	S_XX -> XE I XG
	E -> N1 TOUGH INF 
	XE -> IT TOUGH INF
    I -> RC_OBJ COMP V2
	G -> GAP_MARKER ADV GAP_MARKER
	XG -> GAP_MARKER N2 GAP_MARKER ADV
    COMP -> "that"
	N1 -> "the pizza" | "that ball" | "his wallet"
    IT -> "it"
    TOUGH -> "is tough" | "is annoying" | "is simple" | "is embarrassing"
	INF -> "to steal" | "to throw" | "to lift"
    RC_OBJ -> "the box" | "this truck" | "his backpack"
    V2 -> "hauled" | "had" | "transported"
	N2 -> "your jacket" | "a phone" | "the paperwork"
	ADV -> "carefully" | "without falling" | "quickly"
"""
