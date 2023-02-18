# %%
"""This file is for recreating Autodidax"""
from __future__ import annotations
from contextlib import contextmanager

# Primops

from typing import Any, NamedTuple, Sequence
from functools import partial

import numpy as np


class Primitive(NamedTuple):
    name: str


def bind1(op: Primitive, *args, **kwargs):
    (out,) = bind(op, *args, **kwargs)
    return out


# + * neg sub transpose broadcast > <
sin_op = Primitive("sin")
cos_op = Primitive("cos")
add_op = Primitive("add_op")
neg_op = Primitive("neg_op")
sub_op = Primitive("sub_op")
gt_op = Primitive("gt_op")
lt_op = Primitive("lt_op")
transpose_op = Primitive("transpose_op")
broadcast_op = Primitive("broadcast_op")
reduce_sum_op = Primitive("reduce_sum_op")
# complete the pattern


def add(x, y):
    return bind1(add_op, x, y)


def mul(x, y):
    return bind1(mul_op, x, y)


def neg(x):
    # `return mul(-1, x)` is too naive since x may not be an integer. We could set the -1 to additive inverse of multiplicative identity.
    return bind1(neg_op, x)


# kwargs are passed by dict
def transpose(x, perm):
    return bind1(transpose_op, x, perm=perm)


def broadcast(x, shape, axis):
    # this could benefit from positional only args
    return bind1(broadcast_op, x, shape=shape, axis=axis)


def reduce_sum(x, axis: Sequence[int] | int | None = None):
    # by default, reduce along *all* axes
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    elif isinstance(axis, int):
        axis = (axis,)  # promote
    elif isinstance(axis, Sequence):
        axis = tuple(axis)
    else:
        axis = tuple(axis)
    return bind1(reduce_sum_op, x, axis=axis)


def sin(x):
    return bind1(sin_op, x)


def cos(x):
    return bind1(cos_op, x)


def greater(x, y):
    return bind1(gt_op, x, y)


def less(x, y):
    return bind1(lt_op, x, y)


class MainTrace(NamedTuple):
    # height in active interpreter stack
    lvl: int
    # interpreter type
    trace_type: type[Trace]
    # ugly dict to dump complex data into
    global_data: Any | None


ACTIVE_INTERPRETERS: list[MainTrace] = []
dynamic_trace: MainTrace | None = None


@contextmanager
def new_main(trace_type: type[Trace], global_data: Any | None = None):
    lvl = len(ACTIVE_INTERPRETERS)
    main = MainTrace(lvl, trace_type, global_data)
    ACTIVE_INTERPRETERS.append(main)

    try:
        yield main
    finally:
        ACTIVE_INTERPRETERS.pop()
    # ugh global vars
