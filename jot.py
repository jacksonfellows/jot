from collections import namedtuple
from math import prod

import numpy as np

# --------------------------------------------------------------------------------
# Bastardized Pratt parser.

VerbSymbol = namedtuple("VerbSymbol", ["symbol"])
AdverbSymbol = namedtuple("AdverbSymbol", ["symbol"])

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

# Hacky system to modify verbs.

adverb_symbols = ["slash", "doublequote"]

def parse_adverbs(parser, verb):
    if parser.token.name in adverb_symbols:
        verb = (parser.expr(40), verb) # ?
    return verb

def unary_verb_token(name, prec):
    def nud(parser):
        return (parse_adverbs(parser, VerbSymbol(name)), parser.expr(prec))
    return nud

def binary_verb_token(name, prec):
    def led(parser, left):
        return (parse_adverbs(parser, VerbSymbol(name)), left, parser.expr(prec))
    return led

end_token = Token(lbp=0, name="end")

constant_tokens = {
    "+": Token(
        lbp=10,
        nud=unary_verb_token("+", 30),
        led=binary_verb_token("+", 10),
    ),
    "-": Token(
        lbp=10,
        nud=unary_verb_token("-", 30),
        led=binary_verb_token("-", 10),
    ),
    "*": Token(
        lbp=20,
        nud=unary_verb_token("*", 30),
        led=binary_verb_token("*", 20),
    ),
    "(": Token(
        lbp=0,
        nud=lparen_nud
    ),
    ")": Token(
        name="rparen",
        lbp=0,
    ),
    "[": Token(
        lbp=0,
        nud=lbracket_nud
    ),
    "]": Token(
        name="rbracket",
        lbp=0
    ),
    "/": Token(
        name="slash",
        lbp=0,
        nud=lambda parser: AdverbSymbol("slash")
    ),
    "\"": Token(
        name="doublequote",
        lbp=0,
    ),
    ">.": Token(
        lbp=5,
        nud=unary_verb_token(">.", 20),
        led=binary_verb_token(">.", 5)
    ),
    "<.": Token(
        lbp=5,
        nud=unary_verb_token("<.", 20),
        led=binary_verb_token("<.", 5)
    ),
    "i.": Token(
        nud=unary_verb_token("i.", 20),
    )
}

token_chars = set("".join(constant_tokens.keys()))

def literal_token(val):
    return Token(
        lbp=0,                  # ?
        nud=lambda parser: val,
    )

class Tokenizer:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def is_num_char(self):
        return self.i < len(self.s) and self.s[self.i] in "0123456789."

    def is_space(self):
        return self.s[self.i].isspace()

    def is_token_char(self):
        return self.s[self.i] in token_chars

    def consume(self):
        if self.i >= len(self.s):
            return end_token

        while self.is_space(): self.i += 1

        if self.is_num_char():
            start_i = self.i
            while self.is_num_char(): self.i += 1
            return literal_token(np.array(float(self.s[start_i:self.i])))
        else:
            start_i = self.i
            while self.is_token_char():
                self.i += 1
                name = self.s[start_i:self.i]
                if name in constant_tokens:
                    return constant_tokens[name]
            assert 0

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

def parse_expr(s):
    p = Parser(s)
    expr = p.parse_expr()
    if p.token.name != "end":
        raise ParseError("Could not parse to end of input.")
    return expr

# --------------------------------------------------------------------------------
# Crude evaluation.

class EvalError(ValueError):
    pass

Verb = namedtuple("Verb", ["symbol", "name", "unary_rank", "binary_rank", "unary_ufunc", "binary_ufunc", "unary_func", "binary_func"], defaults=[None]*8)

def integers_func(shape):
    # TODO handle negative shapes like in J?
    ishape = tuple(int(s) for s in shape)
    if ishape != tuple(shape):
        raise EvalError(f"Invalid shape for integers: {shape}")
    count = prod(ishape)
    return np.arange(count).reshape(ishape)

verbs = [
    Verb(symbol="+", unary_rank=0, unary_ufunc=np.conjugate, binary_rank=0, binary_ufunc=np.add),
    Verb(symbol="-", unary_rank=0, unary_ufunc=np.negative, binary_rank=0, binary_ufunc=np.subtract),
    Verb(symbol=">.", unary_rank=0, unary_ufunc=np.ceil, binary_rank=0, binary_ufunc=np.maximum),
    Verb(symbol="<.", unary_rank=0, unary_ufunc=np.floor, binary_rank=0, binary_ufunc=np.minimum),
    Verb(symbol="*", unary_rank=0, unary_ufunc=np.sign, binary_rank=0, binary_ufunc=np.multiply),
    Verb(symbol="i.", unary_rank=1, unary_func=integers_func)
]
symbol_to_verb = {VerbSymbol(v.symbol): v for v in verbs}

def apply_ranked_verb_unary(verb: Verb, noun):
    rank = verb.unary_rank
    if rank < 0: rank = len(noun.shape) + 1 + rank
    if rank == 1 and len(noun.shape) < 1:
        noun = noun.reshape(1)
    if rank == len(noun.shape):
        return verb.unary_func(noun)
    raise EvalError("Not yet supported!")

def apply_ranked_verb_binary(verb: Verb, noun1, noun2):
    rank = verb.binary_rank
    if rank == float("inf"):
        return verb.binary_func(noun1, noun2)
    raise EvalError("Not yet supported!")

def apply_ranked_verb(verb: Verb, nouns):
    if len(nouns) == 1:
        return apply_ranked_verb_unary(verb, nouns[0])
    return apply_ranked_verb_binary(verb, nouns[0], nouns[1])

def eval_verb(verb: Verb, nouns):
    if len(nouns) == 1 and verb.unary_rank == 0 and verb.unary_ufunc is not None:
        return verb.unary_ufunc(*nouns)
    if len(nouns) == 2 and verb.binary_rank == 0 and verb.binary_ufunc is not None:
        return verb.binary_ufunc(*nouns)
    else:
        return apply_ranked_verb(verb, nouns)

def eval_adverb(adverb_symbol: AdverbSymbol, verb_symbol: VerbSymbol):
    if adverb_symbol.symbol == "slash":
        verb = symbol_to_verb[verb_symbol]
        ufunc = verb.binary_ufunc
        return Verb(
            unary_rank=-1,
            unary_func=lambda x: ufunc.reduce(x, axis=-1),
            binary_rank=float("inf"),
            binary_func=lambda x, y: ufunc.outer(x, y)
        )
    else:
        raise EvalError(f"Unrecognized adverb `{adverb_symbol.symbol}`.")

def eval_expr(expr):
    print(expr)
    if type(expr) == np.ndarray:
        return expr
    if type(expr[0]) == tuple:
        return eval_expr((eval_expr(expr[0]), *expr[1:]))
    if expr[0] == "array":
        return np.array([eval_expr(e) for e in expr[1]])
    if type(expr[0]) == VerbSymbol:
        return eval_expr((symbol_to_verb[expr[0]], *expr[1:]))
    if type(expr[0]) == Verb:
        return eval_verb(expr[0], [eval_expr(x) for x in expr[1:]])
    if type(expr[0]) == AdverbSymbol:
        return eval_adverb(expr[0], expr[1])

def repl():
    while 1:
        try:
            inp = input("> ")
            try:
                print(eval_expr(parse_expr(inp)))
            except Exception as e:
                print("Error:", e)
        except EOFError:
            break
