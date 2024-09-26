from collections import namedtuple

import numpy as np

# --------------------------------------------------------------------------------
# Bastardized Pratt parser.

class ParseError(ValueError):
    pass

Token = namedtuple("Token", ["lbp", "led", "nud", "name"], defaults=[None]*4)

def lparen_nud(parser):
    e = parser.expr(0)
    if parser.token.name != "rparen":
        raise ParseError("Expecting closing ).")
    parser.token = parser.next()
    return e

def lbracket_nud(parser):
    es = []
    while parser.token.name != "rbracket":
         # rbp=10 allows negatives in array literals as expected.
        es.append(parser.expr(10))
    parser.token = parser.next()
    return ("array", es)

constant_tokens = {
    "end": Token(lbp=0),
    "add": Token(
        lbp=10,
        nud=lambda parser: ("unary_add", parser.expr(20)),
        led=lambda parser, left: ("add", left, parser.expr(10)),
    ),
    "sub": Token(
        lbp=10,
        nud=lambda parser: ("unary_sub", parser.expr(20)),
        led=lambda parser, left: ("sub", left, parser.expr(10)),
    ),
    "lparen": Token(
        lbp=0,
        nud=lparen_nud
    ),
    "rparen": Token(
        name="rparen",
        lbp=0,
    ),
    "lbracket": Token(
        lbp=0,
        nud=lbracket_nud
    ),
    "rbracket": Token(
        name="rbracket",
        lbp=0
    )
}

def literal_token(val):
    return Token(
        lbp=0,                  # ?
        nud=lambda expr: val,
    )

token_name_lookup = {
    "+": "add",
    "-": "sub",
    "(": "lparen",
    ")": "rparen",
    "[": "lbracket",
    "]": "rbracket"
}

class Tokenizer:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def is_num_char(self):
        return self.i < len(self.s) and self.s[self.i] in "0123456789."

    def is_space(self):
        return self.s[self.i].isspace()

    def consume(self):
        if self.i >= len(self.s):
            return constant_tokens["end"]

        while self.is_space(): self.i += 1

        if self.is_num_char():
            start_i = self.i
            while self.is_num_char(): self.i += 1
            return literal_token(float(self.s[start_i:self.i]))
        else:
            if self.s[self.i] in token_name_lookup:
                start_i = self.i
                self.i += 1
                return constant_tokens[token_name_lookup[self.s[start_i:self.i]]]

class Parser:
    def __init__(self, s):
        self.next = Tokenizer(s).consume
        self.token = self.next()

    def parse_expr(self):
        return self.expr(0)

    def expr(self, rbp):
        t = self.token
        self.token = self.next()
        left = t.nud(self)
        while rbp < self.token.lbp:
            t = self.token
            self.token = self.next()
            left = t.led(self, left)
        return left

# --------------------------------------------------------------------------------
# Crude evaluation.

Verb = namedtuple("Verb", ["name", "nin", "nout", "rank"])

verbs = [
    Verb(name="add", nin=2, nout=1, rank=0),
    Verb(name="unary_add", nin=1, nout=1, rank=0),
    Verb(name="sub", nin=2, nout=1, rank=0),
    Verb(name="unary_sub", nin=1, nout=1, rank=0)
]
name_to_verb = {v.name: v for v in verbs}

verb_to_ufunc = {
    "add": np.add,
    "unary_add": lambda _: _,
    "sub": np.subtract,
    "unary_sub": np.negative
}

def eval_verb(verb: Verb, nouns):
    assert verb.nin == len(nouns)
    return verb_to_ufunc[verb.name](*nouns)

def eval_expr(expr):
    if type(expr) == np.ndarray or type(expr) == float:
        return expr

    if expr[0] == "array":
        return np.array([eval_expr(e) for e in expr[1]])

    verb = name_to_verb[expr[0]]
    return eval_verb(verb, [eval_expr(e) for e in expr[1:]])
