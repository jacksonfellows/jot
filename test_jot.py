import numpy as np

from jot import eval_expr, parse_expr


def test_basic():
    assert eval_expr(parse_expr("2 + --2")) == 4
    assert np.all(eval_expr(parse_expr("[1 2 3] - 1")) == np.array([0, 1, 2]))
    assert np.all(eval_expr(parse_expr("[-1 2] + [3 -4]")) == np.array([2, -2]))
