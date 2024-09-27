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

def parse_adverb(parser):
    if parser.token.name == "slash":
        parser.token = parser.next()
        return AdverbSymbol("slash")

def unary_verb_token(name, prec):
    def nud(parser):
        adverb = parse_adverb(parser)
        if adverb:
            return (adverb, VerbSymbol(name), parser.expr(prec))
        else:
            return (VerbSymbol(name), parser.expr(prec))
    return nud

def binary_verb_token(name, prec):
    def led(parser, left):
        adverb = parse_adverb(parser)
        if adverb:
            return (adverb, VerbSymbol(name), left, parser.expr(prec))
        else:
            return (VerbSymbol(name), left, parser.expr(prec))
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
        lbp=0
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
        nud=lambda expr: val,
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

Verb = namedtuple("Verb", ["symbol", "name", "nin", "nout", "rank"])

verbs = [
    Verb(symbol="+", name="plus", nin=2, nout=1, rank=0),
    Verb(symbol="+", name="conjugate", nin=1, nout=1, rank=0),
    Verb(symbol="-", name="minus", nin=2, nout=1, rank=0),
    Verb(symbol="-", name="negate", nin=1, nout=1, rank=0),
    Verb(symbol=">.", name="ceiling", nin=1, nout=1, rank=0),
    Verb(symbol=">.", name="max", nin=2, nout=1, rank=0),
    Verb(symbol="<.", name="floor", nin=1, nout=1, rank=0),
    Verb(symbol="<.", name="min", nin=2, nout=1, rank=0),
    Verb(symbol="*", name="sign", nin=1, nout=1, rank=0),
    Verb(symbol="*", name="times", nin=2, nout=1, rank=0),
    Verb(symbol="i.", name="integers", nin=1, nout=1, rank=1)
]
symbol_to_unary_verb = {VerbSymbol(v.symbol): v for v in verbs if v.nin == 1}
symbol_to_binary_verb = {VerbSymbol(v.symbol): v for v in verbs if v.nin == 2}

verb_name_to_ufunc = dict(
    plus=np.add,
    conjugate=np.conjugate,
    minus=np.subtract,
    negate=np.negative,
    ceiling=np.ceil,
    floor=np.floor,
    max=np.maximum,
    min=np.minimum,
    sign=np.sign,
    times=np.multiply,
)

def integers_func(shape):
    # TODO handle negative shapes like in J?
    ishape = tuple(int(s) for s in shape)
    if ishape != tuple(shape):
        raise EvalError(f"Invalid shape for integers: {shape}")
    count = prod(ishape)
    return np.arange(count).reshape(ishape)

verb_name_to_func = dict(
    integers=integers_func
)

def apply_ranked_verb(verb: Verb, nouns):
    assert len(nouns) == 1      # Support binary verbs later.
    noun = nouns[0]
    if verb.rank == 1 and len(noun.shape) < 1:
        noun = noun.reshape(1)
    if verb.rank == len(noun.shape):
        return verb_name_to_func[verb.name](noun)
    raise EvalError("Not yet supported!")

def eval_verb(symbol: VerbSymbol, nouns):
    if len(nouns) == 1:
        verb = symbol_to_unary_verb[symbol]
    elif len(nouns) == 2:
        verb = symbol_to_binary_verb[symbol]
    else:
        assert 0
    if verb.rank == 0:
        # Can apply rank-0 verbs as numpy ufuncs.
        return verb_name_to_ufunc[verb.name](*nouns)
    else:
        return apply_ranked_verb(verb, nouns)

def eval_adverb(adverb_symbol: AdverbSymbol, verb_symbol: VerbSymbol, nouns):
    if adverb_symbol.symbol == "slash":
        if len(nouns) == 1:
            # insert
            rank = -1
            verb = symbol_to_binary_verb[verb_symbol]
            ufunc = verb_name_to_ufunc[verb.name]
            return ufunc.reduce(nouns[0], axis=rank)
        elif len(nouns) == 2:
            # table
            verb = symbol_to_binary_verb[verb_symbol]
            ufunc = verb_name_to_ufunc[verb.name]
            return ufunc.outer(nouns[0], nouns[1])
        else:
            assert 0
    else:
        raise EvalError(f"Unrecognized adverb `{adverb_symbol.symbol}`.")

def eval_expr(expr):
    if type(expr) == np.ndarray:
        return expr
    if expr[0] == "array":
        return np.array([eval_expr(e) for e in expr[1]])
    if type(expr[0]) == VerbSymbol:
        return eval_verb(expr[0], [eval_expr(e) for e in expr[1:]])
    if type(expr[0]) == AdverbSymbol:
        # Do I need to eval verb slot?
        return eval_adverb(expr[0], expr[1], [eval_expr(e) for e in expr[2:]])

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
