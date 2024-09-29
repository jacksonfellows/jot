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
    (">. 2.1", 3),
    (">./ [3 5 -3 4]", 5),
    ("<. 2.1", 2),
    ("<./ [3 5 -3 4]", -3),
    ("*-2", -1),
    ("*2", 1),
    ("[1 2 3 4] */ [1 2 3 4]", np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]])),
    ("1 + 2*2 + 1", 6),
    ("1 + 2*(2 + 1)", 7),
    ("i. [2 10]", np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])),
    ("+/\"1 [[1 2 3] [1 2 3] [1 2 3]]", np.array([6, 6, 6])),
    ("+/\"2 [[1 2 3] [1 2 3] [1 2 3]]", np.array([3, 6, 9])),
    ("-/\"2 [[1 2 3] [1 2 3]]", np.zeros(3)),
    ("-/\"1 [[1 2 3] [1 2 3]]", np.array([-4, -4])),
    ("i.10", np.arange(10)),
    ("[0 1]+i.[2 3]", np.array([[0,1,2],[4,5,6]]))
]

def test_basic():
    for s, a in test_cases:
        print(f"{s} => {a}")
        assert np.all(eval_expr(parse_expr(s)) == a)
