import numpy as np

from jot import eval_expr, parse_expr


def test_basic():
    assert eval_expr(parse_expr("2 + --2")) == 4
    assert np.all(eval_expr(parse_expr("[1 2 3] - 1")) == np.array([0, 1, 2]))
    assert np.all(eval_expr(parse_expr("[-1 2] + [3 -4]")) == np.array([2, -2]))

def test_insert():
    assert eval_expr(parse_expr("+/ [1 2 3 4 5]")) == 15

    # Different from J but makes sense to me.
    assert eval_expr(parse_expr("-/ [1 2 3 4 5]")) == -13

def test_table():
    assert np.all(eval_expr(parse_expr("[0 1 2 3] +/ [0 1 2 3]")) == np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
    assert np.all(eval_expr(parse_expr("[0 1 2 3 4 5 6 7 8 9] -/ [0 1]")) == np.array([[0, -1], [1,  0], [2,  1], [3,  2], [4,  3], [5,  4], [6,  5], [7,  6], [8,  7], [9,  8]]))
