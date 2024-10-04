import itertools
from collections import namedtuple
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

INF = float("inf")

# --------------------------------------------------------------------------------
# Tokenizer.

VerbToken = namedtuple("VerbToken", ["symbol"])
AdverbToken = namedtuple("AdverbToken", ["symbol"])
ConjunctionToken = namedtuple("ConjunctionToken", ["symbol"])

END_TOKEN = "END_TOKEN"
LEFT_PAREN_TOKEN = "LEFT_PAREN_TOKEN"
RIGHT_PAREN_TOKEN = "RIGHT_PAREN_TOKEN"
LEFT_BRACKET_TOKEN = "LEFT_BRACKET_TOKEN"
RIGHT_BRACKET_TOKEN = "RIGHT_BRACKET_TOKEN"

class ParseError(ValueError):
    pass

verb_tokens = set(["+", "-", "*", "i.", "<.", ">.", "=", "$", ".", "|", "?"])
adverb_tokens = set(["/", "~"])
# " is an adverb but is treated like a conjunction bc. of how the parser works.
conjunction_tokens = set(["@", "\""])

binary_verb_prec_levels = (
    ("$"),
    ("*", "."),
    ("+", "-"),
    (">.", "<.", "=")
)

TRANSPOSE_TOKEN = "TRANSPOSE_TOKEN"

token_chars = "".join(verb_tokens | adverb_tokens | conjunction_tokens | set("'"))

LiteralToken = namedtuple("LiteralToken", ["value"])

class Tokenizer:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def curr(self):
        return self.s[self.i]

    def is_num_char(self):
        return self.i < len(self.s) and self.s[self.i] in "0123456789."

    def is_space(self):
        return self.s[self.i].isspace()

    def is_token_char(self):
        return self.s[self.i] in token_chars

    def consume(self):
        if self.i >= len(self.s):
            return END_TOKEN

        while self.is_space(): self.i += 1

        char_to_token = {
            "∞": LiteralToken(np.array(INF)),
            "(": LEFT_PAREN_TOKEN,
            ")": RIGHT_PAREN_TOKEN,
            "[": LEFT_BRACKET_TOKEN,
            "]": RIGHT_BRACKET_TOKEN
        }

        if self.curr() in char_to_token:
            x = char_to_token[self.curr()]
            self.i += 1
            return x

        if self.curr().isnumeric():
            start_i = self.i
            while self.is_num_char(): self.i += 1
            return LiteralToken(np.array(float(self.s[start_i:self.i])))

        start_i = self.i
        while self.is_token_char():
            self.i += 1
            name = self.s[start_i:self.i]
            if name in verb_tokens:
                return VerbToken(name)
            if name in adverb_tokens:
                return AdverbToken(name)
            if name in conjunction_tokens:
                return ConjunctionToken(name)
            if name == "'":
                return TRANSPOSE_TOKEN
        assert 0

# --------------------------------------------------------------------------------
# Parser.

def source_verb(expr):
    if type(expr) == VerbToken:
        return expr.symbol
    if type(expr) == str:
        return expr
    assert type(expr) == tuple
    if expr[0] == "apply_adverb":
        return source_verb(expr[2])
    if expr[0] == "apply_conjunction":
        if expr[1] == ConjunctionToken("\""):
            return source_verb(expr[2])
        return "+"              # Fake reasonable precedence.

class Parser:
    def __init__(self, s):
        self.next_token = Tokenizer(s).consume
        self.current_token = self.next_token()

    def consume(self):
        x = self.current_token
        self.current_token = self.next_token()
        return x

    def parse_expr(self, array_literal=False):
        atoms = []

        if array_literal:
            while self.current_token != RIGHT_BRACKET_TOKEN:
                atoms.append(self.parse_atom())
        else:
            atom = self.parse_atom()
            while atom is not None:
                atoms.append(atom)
                atom = self.parse_atom()

        # Transpose (special case).
        i = 0
        while i < len(atoms):
            if atoms[i] == TRANSPOSE_TOKEN:
                atoms[i-1:i+1] = [("apply_verb", "'", atoms[i-1])]
            else:
                i += 1
        to_symbol = lambda x: x.symbol if type(x) == VerbToken else x
        # Adverbs.
        i = 0
        while i < len(atoms):
            if type(atoms[i]) == AdverbToken:
                atoms[i-1:i+1] = [("apply_adverb", atoms[i].symbol, to_symbol(atoms[i-1]))]
            else:
                i += 1
        # Conjunctions.
        i = 0
        while i < len(atoms):
            if type(atoms[i]) == ConjunctionToken:
                atoms[i-1:i+2] = [("apply_conjunction", atoms[i].symbol, to_symbol(atoms[i-1]), to_symbol(atoms[i+1]))]
            else:
                i += 1
        # Verbs.
        is_verb = lambda x: type(x) == VerbToken or (type(x) == tuple and x[0] in ("apply_adverb", "apply_conjunction"))
        # First, handle unary verbs.
        i = len(atoms)-1
        while len(atoms) > 1 and i >= 0:
            if is_verb(atoms[i]) and (i == 0 or is_verb(atoms[i-1])):
                atoms[i:i+2] = [("apply_verb", to_symbol(atoms[i]), atoms[i+1])]
            else:
                i -= 1
        # Next, handle binary verbs (respecting precedence).
        for verbs in binary_verb_prec_levels:
            i = len(atoms)-2
            while len(atoms) > 1 and i >= 1:
                if is_verb(atoms[i]) and source_verb(atoms[i]) in verbs:
                    atoms[i-1:i+2] = [("apply_verb", to_symbol(atoms[i]), atoms[i-1], atoms[i+1])]
                i -= 1

        if array_literal:
            return atoms
        else:
            if len(atoms) != 1: raise ParseError()
            return atoms[0]

    def parse_atom(self):
        if type(self.current_token) == LiteralToken:
            return self.consume().value
        if self.current_token == LEFT_PAREN_TOKEN:
            return self.parse_parens()
        if self.current_token == LEFT_BRACKET_TOKEN:
            return self.parse_array_literal()
        if type(self.current_token) in (VerbToken, AdverbToken, ConjunctionToken):
            return self.consume()
        if self.current_token == TRANSPOSE_TOKEN:
            return self.consume()

    def parse_parens(self):
        assert self.current_token == LEFT_PAREN_TOKEN
        self.consume()
        x = self.parse_expr()
        assert self.current_token == RIGHT_PAREN_TOKEN
        self.consume()
        return x

    def parse_array_literal(self):
        assert self.current_token == LEFT_BRACKET_TOKEN
        self.consume()
        arr = self.parse_expr(array_literal=True)
        assert self.current_token == RIGHT_BRACKET_TOKEN
        self.consume()
        return ("array_literal", *arr)

def parse_expr(s):
    parser = Parser(s)
    x = parser.parse_expr()
    assert parser.current_token == END_TOKEN
    return x

# --------------------------------------------------------------------------------
# Evaluation.

def apply_as_rank_unary(x, rank, ufunc):
    # Special-case numpy ufuncs:
    if rank == 0 and type(ufunc) in (np.ufunc, np.vectorize):
        return ufunc(x)

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

    # Special-case numpy ufuncs:
    if rank1 == 0 and rank2 == 0 and type(bfunc) in (np.ufunc, np.vectorize):
        return bfunc(x, y)

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

Verb = namedtuple("Verb", ["symbol", "urank", "ufunc", "brank1", "brank2", "bfunc"])

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
    Verb(symbol="'", urank=INF, ufunc=np.transpose, brank1=None, brank2=None, bfunc=None),
    Verb(symbol=".", urank=None, ufunc=None, brank1=INF, brank2=INF, bfunc=np.dot),
    Verb(symbol="|", urank=0, ufunc=np.abs, brank1=None, brank2=None, bfunc=None),
]
symbol_to_verb = {v.symbol: v for v in verbs}

# Make sure I didn't miss a verb in tokenizer.
assert (set([v.symbol for v in verbs]) - set("'")).issubset(verb_tokens)

def eval_verb(verb, *args):
    if len(args) == 1:
        return apply_as_rank_unary(args[0], verb.urank, verb.ufunc)
    if len(args) == 2:
        return apply_as_rank_binary(args[0], args[1], verb.brank1, verb.brank2, verb.bfunc)
    raise EvalError(f"Too many nouns for verb {verb_symbol}.")

def eval_slash(verb):
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

def eval_tilde(verb):
    return Verb(
        symbol=None,
        urank=INF,
        ufunc=lambda x: eval_verb(verb, x, x),
        brank1=None,
        brank2=None,
        bfunc=None
    )

def eval_at(u, v):
    return Verb(
        symbol=None,
        urank=v.urank,
        ufunc=lambda x: u.ufunc(v.ufunc(x)),
        brank1=v.brank1,
        brank2=v.brank2,
        bfunc=lambda x, y: u.ufunc(v.bfunc(x, y))
    )

def eval_rank(verb, rank):
    assert type(rank) == np.ndarray
    if len(rank.shape) == 0:
        urank = brank1 = brank2 = rank
    elif len(rank.shape) == 1:
        if len(rank) == 1:
            urank = brank1 = brank2 = rank[0]
        elif len(rank) == 2:
            brank1, brank2 = rank
            urank = brank2
        elif len(rank) == 3:
            urank, brank1, brank2 = rank

    return Verb(
        symbol=None,
        urank=urank,
        ufunc=lambda x: eval_verb(verb, x),
        brank1=brank1,
        brank2=brank2,
        bfunc=lambda x,y: eval_verb(verb, x, y),
    )

modifiers = {
    "/": eval_slash,
    "~": eval_tilde,
    "@": eval_at,
    "\"": eval_rank
}

def eval_expr(expr):
    if type(expr) == tuple:
        args = [eval_expr(x) for x in expr[1:]]
        if expr[0] == "array_literal":
            return np.array(args)
        if expr[0] == "apply_verb":
            return eval_verb(*args)
        if expr[0] in ("apply_adverb", "apply_conjunction"):
            return modifiers[args[0]](*args[1:])
    elif type(expr) == str and expr in symbol_to_verb:
        return symbol_to_verb[expr]
    else:
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
