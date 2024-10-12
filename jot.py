import itertools
from collections import namedtuple
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
import scipy
from numpy.lib.stride_tricks import sliding_window_view

INF = float("inf")

SPEEDUP = True

# --------------------------------------------------------------------------------
# State (for parsing and evaluation).

class JotState:
    def __init__(self, vars=None):
        if vars is None:
            self.vars = {}
        else:
            self.vars = vars

    def assign(self, name, value):
        if name in symbol_to_verb:
            raise EvalError(f"Cannot assign to {name}")
        self.vars[name] = value

    def lookup(self, name):
        if name in symbol_to_verb:
            return symbol_to_verb[name]
        return self.vars.get(name)

    def lookup_type(self, name):
        x = self.lookup(name)
        if x is None:
            return "undef"
        if type(x) is Verb:
            return "verb"
        return "noun"

    def binary_verb_prec_levels(self):
        user_verbs = tuple(v.symbol for v in self.vars.values() if type(v) == Verb)
        return tuple(level + user_verbs if USER_FAKE_PREC in level else level for level in binary_verb_prec_levels)

    def bind(self, names, vals):
        new_state = JotState(self.vars.copy())
        for name,val in zip(names, vals):
            new_state.assign(name, val)
        return new_state

# --------------------------------------------------------------------------------
# Tokenizer.

VerbToken = namedtuple("VerbToken", ["symbol"])
AdverbToken = namedtuple("AdverbToken", ["symbol"])
ConjunctionToken = namedtuple("ConjunctionToken", ["symbol"])
NounToken = namedtuple("NounToken", ["symbol"])
UndefToken = namedtuple("UndefToken", ["symbol"])

END_TOKEN = "END_TOKEN"
LEFT_PAREN_TOKEN = "LEFT_PAREN_TOKEN"
RIGHT_PAREN_TOKEN = "RIGHT_PAREN_TOKEN"
LEFT_BRACKET_TOKEN = "LEFT_BRACKET_TOKEN"
RIGHT_BRACKET_TOKEN = "RIGHT_BRACKET_TOKEN"
COMMA_TOKEN = "COMMA_TOKEN"
LAMBDA_TOKEN = "LAMBDA_TOKEN"
COLON_TOKEN = "COLON_TOKEN"
ASSIGN_TOKEN = "ASSIGN_TOKEN"

class ParseError(ValueError):
    pass

verb_tokens = set(["+", "-", "*", "i.", "<.", ">.", "=", "$", ".", "|", "?", "sin", "cos", "tan", "asin", "acos", "atan", "√", "<", ">", "÷", "!", "#"])
adverb_tokens = set(["/", "~", "\\"])
# " is an adverb but is treated like a conjunction bc. of how the parser works.
conjunction_tokens = set(["@", "\"", "@:"])

binary_verb_prec_levels = (
    ("$"),
    ("*", ".", "÷", "!"),
    ("+", "-", "atan"),
    (">.", "<.", "=", ">", "<")
)
USER_FAKE_PREC = "*"

TRANSPOSE_TOKEN = "TRANSPOSE_TOKEN"

token_chars = "".join(verb_tokens | adverb_tokens | conjunction_tokens | set("'"))

LiteralToken = namedtuple("LiteralToken", ["value"])

class Tokenizer:
    def __init__(self, s, state: JotState):
        self.s = s
        self.i = 0
        self.state = state

    def curr(self):
        return self.s[self.i]

    def is_num_char(self):
        return self.i < len(self.s) and self.s[self.i] in "0123456789."

    def is_space(self):
        return self.s[self.i].isspace()

    def is_token_char(self):
        return self.i < len(self.s) and self.s[self.i] in token_chars

    def is_alpha(self):
        return self.i < len(self.s) and self.curr().isalpha()

    def consume(self):
        if self.i >= len(self.s):
            return END_TOKEN

        while self.is_space(): self.i += 1

        char_to_token = {
            "∞": LiteralToken(np.array(INF)),
            "π": LiteralToken(np.array(np.pi)), # Right unicode symbol?
            "(": LEFT_PAREN_TOKEN,
            ")": RIGHT_PAREN_TOKEN,
            "[": LEFT_BRACKET_TOKEN,
            "]": RIGHT_BRACKET_TOKEN,
            "←": ASSIGN_TOKEN,
            ",": COMMA_TOKEN,
            "λ": LAMBDA_TOKEN,
            ":": COLON_TOKEN,
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
            if self.i+1 < len(self.s) and any(name + self.s[self.i] in t for t in (verb_tokens, adverb_tokens, conjunction_tokens)):
                continue
            if name in verb_tokens:
                return VerbToken(name)
            if name in adverb_tokens:
                return AdverbToken(name)
            if name in conjunction_tokens:
                return ConjunctionToken(name)
            if name == "'":
                return TRANSPOSE_TOKEN

        self.i = start_i        # Reset if failed to parse known token.
        while self.is_alpha():
            self.i += 1
        name = self.s[start_i:self.i]
        type = self.state.lookup_type(name)
        return {"undef": UndefToken, "verb": VerbToken, "noun": NounToken}[type](name)

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
    return USER_FAKE_PREC   # Fake precedence.


class Parser:
    def __init__(self, s, state: JotState):
        self.state = state
        self.next_token = Tokenizer(s, self.state).consume
        self.token_stack = []
        self.current_token = self.next_token()

    def consume(self):
        x = self.current_token
        if len(self.token_stack) == 0:
            self.current_token = self.next_token()
        else:
            self.current_token = self.token_stack.pop()
        return x

    def peek(self):
        x = self.next_token()
        self.token_stack.append(x)
        return x

    def parse_line(self):
        if self.peek() == ASSIGN_TOKEN:
            # Assignment.
            name = self.consume()
            assert self.consume() == ASSIGN_TOKEN
            val = self.parse_expr()
            return ("assign", name, val)
        else:
            return self.parse_expr()

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
        is_verb = lambda x: type(x) == VerbToken or (type(x) == tuple and x[0] in ("apply_adverb", "apply_conjunction", "make_train", "lambda"))
        # First, handle unary verbs.
        i = len(atoms)-2
        while len(atoms) > 1 and i >= 0:
            if is_verb(atoms[i]) and (i == 0 or is_verb(atoms[i-1])) and (not is_verb(atoms[i+1])):
                atoms[i:i+2] = [("apply_verb", to_symbol(atoms[i]), atoms[i+1])]
            else:
                i -= 1
        # Next, handle binary verbs (respecting precedence).
        for verbs in self.state.binary_verb_prec_levels():
            i = len(atoms)-2
            while len(atoms) > 1 and i >= 1:
                if is_verb(atoms[i]) and source_verb(atoms[i]) in verbs and (not is_verb(atoms[i-1])) and (not is_verb(atoms[i+1])):
                    atoms[i-1:i+2] = [("apply_verb", to_symbol(atoms[i]), atoms[i-1], atoms[i+1])]
                i -= 1

        if array_literal:
            return atoms
        else:
            if len(atoms) == 1:
                return atoms[0]
            return ("make_train", *[to_symbol(a) for a in atoms])

    def parse_atom(self):
        if type(self.current_token) == LiteralToken:
            return self.consume().value
        if self.current_token == LEFT_PAREN_TOKEN:
            return self.parse_parens()
        if self.current_token == LEFT_BRACKET_TOKEN:
            return self.parse_array_literal()
        if type(self.current_token) in (VerbToken, AdverbToken, ConjunctionToken, NounToken, UndefToken):
            return self.consume()
        if self.current_token == TRANSPOSE_TOKEN:
            return self.consume()
        if self.current_token == LAMBDA_TOKEN:
            return self.parse_lambda()

    def parse_lambda(self):
        assert self.consume() == LAMBDA_TOKEN
        vars = []
        while 1:
            var = self.consume()
            vars.append(var.symbol)
            if self.current_token == COLON_TOKEN:
                self.consume()
                break
            if self.current_token == COMMA_TOKEN:
                self.consume()
                pass
            else:
                raise ParseError()
        body = self.parse_expr()
        return ("lambda", vars, body)

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

def parse_expr(s, state: JotState):
    parser = Parser(s, state)
    x = parser.parse_expr()
    assert parser.current_token == END_TOKEN
    return x

def parse_line(s, state: JotState):
    parser = Parser(s, state)
    x = parser.parse_line()
    assert parser.current_token == END_TOKEN
    return x

# --------------------------------------------------------------------------------
# Evaluation.

def apply_as_rank_unary(x, rank, ufunc):
    # Special-case numpy ufuncs:
    if SPEEDUP and rank == 0 and type(ufunc) in (np.ufunc, np.vectorize):
        return ufunc(x)

    if rank == INF:
        return ufunc(x)
    if len(x.shape) <= rank:
        while len(x.shape) < rank: x = x.reshape((1, *x.shape))
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
        while len(y.shape) < len(x.shape): y = y.reshape((1, *y.shape))
        y = np.broadcast_to(y, x.shape)
    elif len(y.shape) > len(x.shape):
        while len(x.shape) < len(y.shape): x = x.reshape((1, *x.shape))
        x = np.broadcast_to(x, y.shape)

    if x.shape != y.shape:
        if prod(x.shape) < prod(y.shape):
            x = np.broadcast_to(x, y.shape)
        else:
            y = np.broadcast_to(y, x.shape)

    # Special-case numpy ufuncs:
    if SPEEDUP and rank1 == 0 and rank2 == 0 and type(bfunc) in (np.ufunc, np.vectorize):
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

@dataclass
class Verb:
    symbol: str
    urank: Any
    ufunc: Any
    brank1: Any
    brank2: Any
    bfunc: Any

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
    Verb(symbol="sin", urank=0, ufunc=np.sin, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="cos", urank=0, ufunc=np.cos, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="tan", urank=0, ufunc=np.tan, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="asin", urank=0, ufunc=np.arcsin, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="acos", urank=0, ufunc=np.arccos, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="atan", urank=0, ufunc=np.arctan, brank1=0, brank2=0, bfunc=np.arctan2),
    Verb(symbol="√", urank=0, ufunc=np.sqrt, brank1=None, brank2=None, bfunc=None),
    Verb(symbol="<", urank=None, ufunc=None, brank1=0, brank2=0, bfunc=np.less),
    Verb(symbol=">", urank=None, ufunc=None, brank1=0, brank2=0, bfunc=np.greater),
    Verb(symbol="÷", urank=0, ufunc=np.reciprocal, brank1=0, brank2=0, bfunc=np.divide),
    Verb(symbol="!", urank=0, ufunc=scipy.special.factorial, brank1=0, brank2=0, bfunc=scipy.special.comb),
    Verb(symbol="#", urank=INF, ufunc=lambda x: np.array(x.shape[0]), brank1=None, brank2=None, bfunc=None),
]
symbol_to_verb = {v.symbol: v for v in verbs}

def eval_verb(verb, *args):
    if len(args) == 1:
        return apply_as_rank_unary(args[0], verb.urank, verb.ufunc)
    if len(args) == 2:
        return apply_as_rank_binary(args[0], args[1], verb.brank1, verb.brank2, verb.bfunc)
    raise EvalError(f"Too many nouns for verb {verb}.")

def reduce_verb(verb, x):
    if len(x.shape) == 0: return x
    func = verb.bfunc
    if SPEEDUP:
        if type(func) == np.ufunc:
            return func.reduce(x, axis=0)
    out = np.zeros(x.shape[1:])
    for j in np.ndindex(x.shape[1:]):
        res = x[0, *j]
        for i in range(1,x.shape[0]):
            res = func(res, x[i, *j])
        out[j] = res
    return out

def outer_verb(verb, x, y):
    func = verb.bfunc
    if SPEEDUP:
        if type(func) == np.ufunc:
            return func.outer(x, y)
    out = np.zeros(x.shape + y.shape)
    for i,j in itertools.product(np.ndindex(x.shape), np.ndindex(y.shape)):
        out[i,j] = func(x[i], x[j])
    return out

def eval_slash(verb):
    return Verb(
        symbol=None,
        urank=INF,
        ufunc=lambda x: reduce_verb(verb, x),
        brank1=INF,
        brank2=INF,
        bfunc=lambda x, y: outer_verb(verb, x, y),
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
    if type(rank) != np.ndarray:
        rank = np.array(rank)
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

def eval_sliding(verb, x):
    L = x.shape[0]
    res = []
    for i in range(1,L+1):
        res.append(eval_verb(verb, x[:i]))
    return np.stack(res)

def eval_sliding2(verb, x, y):
    window = tuple(int(a) for a in x)
    view = sliding_window_view(y, window_shape=window)
    shape = view.shape[:-1]
    i_res = [(i, eval_verb(verb, view[i])) for i in np.ndindex(shape)]
    # Confirm same-shaped results:
    assert all(res.shape == i_res[0][1].shape for _,res in i_res)
    out = np.zeros(shape + i_res[0][1].shape)
    for i,res in i_res:
        out[i] = res
    return out

def eval_bslash(verb):
    return Verb(
        symbol=None,
        urank=INF,
        ufunc=lambda x: eval_sliding(verb, x),
        brank1=1,
        brank2=INF,
        bfunc=lambda x, y: eval_sliding2(verb, x, y)
    )

def eval_colon_at(u, v):
    return Verb(
        symbol=None,
        urank=INF,
        ufunc=lambda x: eval_verb(u, eval_verb(v, x)),
        brank1=INF,
        brank2=INF,
        bfunc=lambda x,y: eval_verb(u, eval_verb(v, x, y))
    )

modifiers = {
    "/": eval_slash,
    "~": eval_tilde,
    "@": eval_at,
    "\"": eval_rank,
    "\\": eval_bslash,
    "@:": eval_colon_at,
}

def make_train(args):
    if len(args) == 3 and all(type(a) == Verb for a in args):
        # Fork
        f, g, h = args
        return Verb(
            symbol=None,
            urank=INF,
            ufunc=lambda x: eval_verb(g, eval_verb(f, x), eval_verb(h, x)),
            brank1=INF,
            brank2=INF,
            bfunc=lambda x, y: eval_verb(g, eval_verb(f, x, y), eval_verb(h, x, y))
        )
    raise EvalError(f"Train with arguments {args} not supported.")

def make_lambda(args, body, current_state):
    if len(args) == 1:
        return Verb(
            symbol=None,
            urank=INF,
            ufunc=lambda x: eval_expr(body, current_state.bind(args, [x])),
            brank1=None,
            brank2=None,
            bfunc=None,
        )
    if len(args) == 2:
        return Verb(
            symbol=None,
            urank=None,
            ufunc=None,
            brank1=INF,
            brank2=INF,
            bfunc=lambda x, y: eval_expr(body, current_state.bind(args, [x, y])),
        )
    raise EvalError("λ has too many arguments.")

def eval_expr(expr, state: JotState):
    if type(expr) == tuple:
        # Don't eval args for these constructs.
        if expr[0] == "lambda":
            return make_lambda(expr[1], expr[2], state)
        if expr[0] == "assign":
            name = expr[1].symbol
            val = eval_expr(expr[2], state)
            if type(val) is Verb:
                val.symbol = name
            state.assign(name, val)
            return val
        args = [eval_expr(x, state) for x in expr[1:]]
        if expr[0] == "array_literal":
            return np.array(args)
        if expr[0] == "apply_verb":
            return eval_verb(*args)
        if expr[0] in ("apply_adverb", "apply_conjunction"):
            return modifiers[args[0]](*args[1:])
        if expr[0] == "make_train":
            return make_train(args)
    if type(expr) == str and expr not in modifiers:
        return state.lookup(expr)
    if type(expr) in (NounToken, UndefToken): # Messed up.
        return state.lookup(expr.symbol)
    return expr

def parse_and_eval(string: str, state: JotState):
    return eval_expr(parse_line(string, state), state)

def repl():
    state = JotState()
    while 1:
        try:
            inp = input("> ")
            try:
                print(parse_and_eval(inp, state))
            except Exception as e:
                print("Error:", e)
        except EOFError:
            break

if __name__ == "__main__":
    repl()
