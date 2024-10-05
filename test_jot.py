import numpy as np

from jot import eval_expr, parse_expr

test_cases = [
    ("2 + --2", 4),
    ("[1 2 3] - 1", np.array([0, 1, 2])),
    ("[(-1) 2] + [3 (-4)]", np.array([2, -2])),
    ("+/ [1 2 3 4 5]", 15),
    ("-/ [1 2 3 4 5]", -13),
    ("[0 1 2 3] +/ [0 1 2 3]", np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])),
    ("[0 1 2 3 4 5 6 7 8 9] -/ [0 1]", np.array([[0, -1], [1,  0], [2,  1], [3,  2], [4,  3], [5,  4], [6,  5], [7,  6], [8,  7], [9,  8]])),
    (">. 2.1", 3),
    (">./ [3 5 (-3) 4]", 5),
    ("<. 2.1", 2),
    ("<./ [3 5 (-3) 4]", -3),
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
    ("[0 1]+i.[2 3]", np.array([[0,1,2],[4,5,6]])),
    ("+~2", 4),
    ("*/~i.10", np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [0, 3, 6, 9, 12, 15, 18, 21, 24, 27], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45], [0, 6, 12, 18, 24, 30, 36, 42, 48, 54], [0, 7, 14, 21, 28, 35, 42, 49, 56, 63], [0, 8, 16, 24, 32, 40, 48, 56, 64, 72], [0, 9, 18, 27, 36, 45, 54, 63, 72, 81]])),
    ("*/~\"1 [[0 1 2] [5 6 7]]", np.array([[[0, 0, 0], [0, 1, 2], [0, 2, 4]], [[25, 30, 35], [30, 36, 42], [35, 42, 49]]])),
    ("$ (i. [1 2 3 4 5])", np.array([1, 2, 3, 4, 5])),
    ("[2 3] $ 0", np.zeros((2, 3))),
    ("[3 3 3] $ [1 2 3]", np.array([[[1, 2, 3,], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])),
    ("=/~i.10", np.eye(10)),
    ("(i.[3 3])'", np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])),
    ("-2*[∞ 4 (-8)]", np.array([-float("inf"), -8, 16])),
    ("+/\"0 [[0 1 2 3] [4 5 6 7]]", np.arange(8).reshape((2,4))),
    ("+/\"1 [[0 1 2 3] [4 5 6 7]]", np.array([6, 22])),
    ("+/\"2 [[0 1 2 3] [4 5 6 7]]", np.array([4, 6, 8, 10])),
    ("+/\"∞ [[0 1 2 3] [4 5 6 7]]", np.array([4, 6, 8, 10])),
    ("i.5 +\"[0 ∞] i.4", np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])),
    ("i.10 +\"[1 1] i.10", 2*np.arange(10)),
    ("[0 (-1) (-2) (-3)]", -np.arange(4)),
    (".~ i.[10 10]", np.arange(100).reshape(10,10)@np.arange(100).reshape(10,10)),
    ("|([10 10]$-1)", np.ones((10,10))),
    ("√.~ i.10", np.linalg.norm(np.arange(10))),
    ("1 atan 1", np.pi/4),
    ("+/\\i.[3 3]", np.array([[0, 1, 2], [3, 5, 7], [9, 12, 15]]))]

equiv_cases = [
    ("(+\"[0 1])~ i.10", "+/~ i.10"),
    ("[2-2 3]", "[0 1+2]"),
    ("[1 0 0] (+/@*) [4 (-5) 8]", "[4 0 0]"),
    ("[0 1 0] (+/@*) [4 (-5) 8]", "[0 (-5) 0]"),
    ("[0 0 1] (+/@*) [4 (-5) 8]", "[0 0 8]"),
    ("[[4 5 6] [6 7 (-4)] [2 3 4]] . [1 0 0]", "[4 6 2]"),
    ("[[4 5 6] [6 7 (-4)] [2 3 4]] . [0 1 0]", "[5 7 3]"),
    ("[[4 5 6] [6 7 (-4)] [2 3 4]] . [0 0 1]", "[6 (-4) 4]"),
    ("| -4", "4"),
    ("| [(-3) 4 5]", "[3 4 5]"),
    ("cos 0", "1"),
    ("cos π", "-1"),
    ("sin 0", "0"),
    ("√81", "9"),
    ("atan (tan 1)", "1"),
    (">./\\[3 4 5 (-5) 4 2 9]", "[3 4 5 5 5 5 9]"),
]

def assert_same(x, y):
    assert x.shape == y.shape
    assert np.all(x == y)

def test_basic():
    for s, a in test_cases:
        print(f"{s} => {a}")
        if type(a) in (int, float):
            a = np.array(a)
        res = eval_expr(parse_expr(s))
        assert_same(a, res)

def test_equiv():
    for a, b in equiv_cases:
        print(f"{a} <=> {b}")
        x = eval_expr(parse_expr(a))
        y = eval_expr(parse_expr(b))
        assert_same(x, y)
