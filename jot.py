import itertools
from collections import namedtuple
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

INF = float("inf")

# --------------------------------------------------------------------------------
# Bastardized Pratt parser.

VerbSymbol = namedtuple("VerbSymbol", ["symbol"])

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
    return (ARRAY_LITERAL_OPERATOR, *es)

# Hacky system to modify verbs.

adverb_symbols = ["/", "\"", "~"]

def parse_adverbs(parser, verb):
    while parser.token.name in adverb_symbols:
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
        name="/",
        lbp=0,
        nud=lambda parser: SLASH_ADVERB_OPERATOR
    ),
    "~": Token(
        name="~",
        lbp=0,
        nud=lambda parser: TILDE_ADVERB_OPERATOR
    ),
    "\"": Token(
        name="\"",
        lbp=0,
        nud=lambda parser: Rank(parser.expr(50)),
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
        lbp=0,                  # ?
        nud=unary_verb_token("i.", 20),
    ),
    "?": Token(
        lbp=0,                  # ?
        nud=unary_verb_token("?", 20)
    ),
    "$": Token(
        lbp=15,
        nud=unary_verb_token("$", 20),
        led=binary_verb_token("$", 15)
    ),
    "=": Token(
        lbp=5,
        nud=unary_verb_token("=", 20),
        led=binary_verb_token("=", 5)
    ),
    "'": Token(
        lbp=30,
        led=lambda parser, left: (parse_adverbs(parser, VerbSymbol("'")), left)
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

        if self.s[self.i] == "∞":
            self.i += 1
            return literal_token(np.array(INF))

        if self.is_num_char():
            start_i = self.i
            while self.is_num_char(): self.i += 1
            return literal_token(np.array(float(self.s[start_i:self.i])))

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

def apply_as_rank_unary(x, rank, ufunc):
    if rank == INF:
        return ufunc(x)
    if len(x.shape) <= rank:
        while len(x.shape) < rank: x = x.reshape((*x.shape, 1))
        return ufunc(x)
    shape = x.shape[:len(x.shape) - int(rank)]
    i_res = [(i, ufunc(x[i])) for i in np.ndindex(shape)]
    # Confirm same-shaped results:
    assert all(res.shape == i_res[0][1].shape for _,res in i_res)
    out = np.zeros(shape + i_res[0][1].shape)
    for i,res in i_res:
        out[i] = res
    return out

def view_as_rank(x, rank):
    if len(x.shape) <= rank:
        while len(x.shape) < rank: x = x.reshape((1, *x.shape))
        yield x
    else:
        shape = x.shape[:len(x.shape) - int(rank)]
        for i in np.ndindex(shape):
            yield x[i]

def apply_as_rank_binary(x, y, rank1, rank2, bfunc):
    if rank1 == INF:
        return apply_as_rank_unary(y, rank2, lambda y: bfunc(x, y))
    if rank2 == INF:
        return apply_as_rank_unary(x, rank1, lambda x: bfunc(x, y))
    # Convert arguments to common shape.
    if len(x.shape) > len(y.shape):
        while len(y.shape) < len(x.shape): y = y.reshape((*y.shape, 1))
        y = np.broadcast_to(y, x.shape)
    elif len(y.shape) > len(x.shape):
        while len(x.shape) < len(y.shape): x = x.reshape((*x.shape, 1))
        x = np.broadcast_to(x, y.shape)
    # Apply verb. Got to be a cleaner way.
    x_iter, y_iter = view_as_rank(x, rank1), view_as_rank(y, rank2)
    shape = x.shape[:len(x.shape) - int(min(rank1, rank2))]
    i_res = [(i, bfunc(x_, y_)) for i,x_,y_ in zip(np.ndindex(shape), itertools.cycle(x_iter), itertools.cycle(y_iter))]
    out = np.zeros(shape + i_res[0][1].shape)
    for i,res in i_res:
        out[i] = res
    return out

class EvalError(ValueError):
    pass

TODO_ERROR = EvalError("Not yet implemented!")

class Operator:
    pass

class ArrayLiteral(Operator):
    def eval(self, args):
        return np.array(args)

    def __repr__(self):
        return "ArrayLiteral()"

ARRAY_LITERAL_OPERATOR = ArrayLiteral()

@dataclass
class Verb(Operator):
    symbol: str
    urank: Any
    brank1: Any
    brank2: Any
    ufunc: Any
    bfunc: Any

    def eval(self, args):
        if len(args) == 1:
            return apply_as_rank_unary(args[0], self.urank, self.ufunc)
        if len(args) == 2:
            return apply_as_rank_binary(args[0], args[1], self.brank1, self.brank2, self.bfunc)
        raise EvalError("Too many nouns for verb {self.symbol}.")

class SlashAdverb(Operator):
    def eval(self, args):
        assert len(args) == 1
        verb: Verb = args[0]
        func = verb.bfunc
        assert type(func) == np.ufunc
        return Verb(
            symbol=None,
            urank=INF,
            ufunc=lambda x: func.reduce(x, axis=0),
            brank1=INF,
            brank2=INF,
            bfunc=lambda x, y: func.outer(x, y)
        )

SLASH_ADVERB_OPERATOR = SlashAdverb()

class TildeAdverb(Operator):
    def eval(self, args):
        assert len(args) == 1
        verb: Verb = args[0]
        return Verb(
            symbol=None,
            urank=INF,
            ufunc=lambda x: verb.eval([x, x]),
            brank1=None,
            brank2=None,
            bfunc=None
        )

TILDE_ADVERB_OPERATOR = TildeAdverb()

@dataclass
class RankedVerb(Operator):
    verb: Verb
    urank: Any
    brank1: Any
    brank2: Any

    def eval(self, args):
        if len(args) == 1:
            return apply_as_rank_unary(args[0], self.urank, lambda x: self.verb.eval([x]))
        elif len(args) == 2:
            return apply_as_rank_binary(args[0], args[1], self.brank1, self.brank2, lambda x,y: self.verb.eval([x,y]))
        else:
            raise EvalError("Too many nouns.")


@dataclass
class Rank(Operator):
    def __init__(self, rank):
        rank = eval_expr(rank)
        assert type(rank) == np.ndarray
        if len(rank.shape) == 0:
            self.urank = self.brank1 = self.brank2 = rank
        elif len(rank.shape) == 1:
            if len(rank) == 1:
                self.urank = self.brank1 = self.brank2 = rank[0]
            elif len(rank) == 2:
                self.brank1, self.brank2 = rank
                self.urank = self.brank2
            elif len(rank) == 3:
                self.urank, self.brank1, self.brank2 = rank
        else:
            assert 0

    def eval(self, args):
        assert len(args) == 1
        verb: Verb = args[0]
        return RankedVerb(verb=verb, urank=self.urank, brank1=self.brank1, brank2=self.brank2)

def integers_func(shape):
    # TODO handle negative shapes like in J?
    ishape = tuple(int(s) for s in shape)
    if ishape != tuple(shape):
        raise EvalError(f"Invalid shape for integers: {shape}")
    count = prod(ishape)
    return np.arange(count).reshape(ishape)

roll_func = np.vectorize(lambda x: np.random.random() if x == 0 else np.random.randint(x))

def shape_bfunc(x, y):
    shape = tuple(int(x_) for x_ in x)
    while len(y.shape) < len(shape): y = y.reshape((1, *y.shape))
    return np.broadcast_to(y, shape)

verbs = [
    Verb(symbol="+", urank=0, ufunc=np.conjugate, brank1=0, brank2=0, bfunc=np.add),
    Verb(symbol="-", urank=0, ufunc=np.negative, brank1=0, brank2=0, bfunc=np.subtract),
    Verb(symbol=">.", urank=0, ufunc=np.ceil, brank1=0, brank2=0, bfunc=np.maximum),
    Verb(symbol="<.", urank=0, ufunc=np.floor, brank1=0, brank2=0, bfunc=np.minimum),
    Verb(symbol="*", urank=0, ufunc=np.sign, brank1=0, brank2=0, bfunc=np.multiply),
    Verb(symbol="i.", urank=1, ufunc=integers_func, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="?", urank=0, ufunc=roll_func, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="$", urank=INF, ufunc=lambda x: np.array(x.shape), brank1=1, brank2=INF, bfunc=shape_bfunc),
    Verb(symbol="=", urank=None, ufunc=None, brank1=0, brank2=0, bfunc=np.equal),
    Verb(symbol="'", urank=INF, ufunc=np.transpose, brank1=None, brank2=None, bfunc=None)
]
symbol_to_verb = {VerbSymbol(v.symbol): v for v in verbs}

def eval_expr(expr):
    if type(expr) == np.ndarray:
        return expr
    if type(expr) == VerbSymbol:
        return symbol_to_verb[expr]

    if type(expr) == tuple:
        head = eval_expr(expr[0])
        tail = [eval_expr(e) for e in expr[1:]]
        assert isinstance(head, Operator)
        return head.eval(tail)

    return expr

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
