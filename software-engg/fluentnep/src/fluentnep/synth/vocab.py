"""Word lists used to template-generate romanized Nepali-English code-mixed
sentences, and the disfluency lexicon injected on top of them.

This is a hand-built substitute for the "scrape YouTube transcripts" step in
the master plan (Week 5). Real transcripts would replace/extend this list
without changing any downstream code.
"""

PRONOUNS = ["ma", "hami", "tapai", "u", "timi", "hamiharu"]

VERBS_NEP = [
    "jaanchhu", "khanchhu", "garchhu", "padhchhu", "herchhu",
    "bhanchhu", "sochchhu", "chahanchhu", "aauchhu", "sutchhu",
]

CONNECTORS = ["ani", "tara", "kinaki", "so", "matlab", "basically", "actually"]

ENGLISH_NOUNS = [
    "school", "exam", "project", "meeting", "assignment", "internet",
    "laptop", "presentation", "deadline", "interview", "semester",
    "college", "app", "server", "dashboard", "dataset",
]

ENGLISH_VERBS = [
    "submit", "finish", "check", "prepare", "review", "deploy",
    "test", "build", "study", "practice",
]

NEP_TIME = ["aaja", "bholi", "haijau", "ahile", "yeta", "tesbela"]

NEP_ADJ = ["ramro", "gaharo", "sajilo", "thulo", "sano"]

SENTENCE_TEMPLATES = [
    "{pron} {time} {eng_noun} {eng_verb} garna {verb_nep}",
    "{pron} {conn} {eng_noun} ko liyera dherai busy {verb_nep}",
    "{time} {pron} {eng_noun} {eng_verb} garnu parne cha",
    "{pron} lai lagcha yo {eng_noun} {adj} cha",
    "{conn} {pron} {eng_noun} {eng_verb} garisakeko {verb_nep}",
    "{pron} {time} college gayera {eng_noun} {eng_verb} garchhu",
    "{pron} sochchhu ki {eng_noun} {adj} hunchha",
    "{conn} hamro {eng_noun} {time} ready hunu parcha",
    "{pron} {eng_noun} ma {eng_verb} garna sakina",
    "tapai lai thaha cha {pron} {eng_noun} {eng_verb} garya",
]

FILLERS = ["uhh", "umm", "matlab", "like", "basically", "err", "aaaa", "haina"]

# Single-word only: every token must be space-delimitable, since tokens/tags
# alignment throughout the pipeline assumes text.split() reconstructs the
# same token list that was labeled.
REPAIR_CONNECTORS = ["matlab", "sorry", "actually"]
