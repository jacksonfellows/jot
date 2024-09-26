import numpy as np

from jot import eval_expr, parse_expr

test_cases = [
    ("2 + --2", 4),
    ("[1 2 3] - 1", np.array([0, 1, 2])),
    ("[-1 2] + [3 -4]", np.array([2, -2])),
    ("+/ [1 2 3 4 5]", 15),
    ("-/ [1 2 3 4 5]", -13),
    ("[0 1 2 3] +/ [0 1 2 3]", np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])),
    ("[0 1 2 3 4 5 6 7 8 9] -/ [0 1]", np.array([[0, -1], [1,  0], [2,  1], [3,  2], [4,  3], [5,  4], [6,  5], [7,  6], [8,  7], [9,  8]])),
]

def test_basic():
    for s, a in test_cases:
        assert np.all(eval_expr(parse_expr(s)) == a)
