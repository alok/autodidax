# %%
import jax
from jax import jit, grad, vmap, pmap
import jax.numpy as jnp
from __future__ import annotations
from typing import NamedTuple
import numpy as np
from contextlib import contextmanager
from typing import Sequence, Any
from dataclasses import dataclass

# %%

# %%
class Primitive(NamedTuple):
    # a wrapper over 
    name: str


add_p, mul_p, neg_p = Primitive("add"), Primitive("mul"), Primitive("neg")
sin_p, cos_p = Primitive("sin"), Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p, less_p = Primitive("greater"), Primitive("less")
transpose_p, broadcast_p = Primitive("transpose"), Primitive("broadcast")


def add(x, y):
    return bind1(add_p, x, y)


def mul(x, y):
    return bind1(mul_p, x, y)


def neg(x):
    return bind1(neg_p, x)


def sin(x):
    return bind1(sin_p, x)


def cos(x):
    return bind1(cos_p, x)


def greater(x, y):
    return bind1(greater_p, x, y)


def less(x, y):
    return bind1(less_p, x, y)

# Here we pass extra arguments with kwargs.
def transpose(x, perm):
    return bind1(transpose_p, x, perm=perm)


def broadcast(x, shape, axes):
    return bind1(broadcast_p, x, shape=shape, axes=axes)


def reduce_sum(x, axis=None):
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    if type(axis) is int:
        axis = (axis,)
    return bind1(reduce_sum_p, x, axis=axis)


def bind1(prim, *args, **params):
    (out,) = bind(prim, *args, **params)
    return out


# %%
class MainTrace(NamedTuple):
    # TODO what does this get me?
    level: int
    # TODO: why a type?
    trace_type: type[Trace]
    # context, probably a big dictionary
    global_data: Any | None


trace_stack: list[MainTrace] = []
dynamic_trace: MainTrace | None = None  # to be employed in Part 3


@contextmanager
def new_main(trace_type: type[Trace], global_data: Any | None = None):
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()


@dataclass
class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val) -> Tracer:
        assert False  # must override

    def lift(self, val) -> Tracer:
        assert False  # must override

    def process_primitive(self, primitive, tracers, params):
        assert False  # must override


class Tracer:
    _trace: Trace

    __array_priority__: int = 1000

    @property
    def aval(self) -> AbstractValue:
        assert False  # must override

    def full_lower(self):
        return self  # default implementation

    def __neg__(self):
        return self.aval._neg(self)

    def __add__(self, other):
        return self.aval._add(self, other)

    def __radd__(self, other):
        return self.aval._radd(self, other)

    def __mul__(self, other):
        return self.aval._mul(self, other)

    def __rmul__(self, other):
        return self.aval._rmul(self, other)

    def __gt__(self, other):
        return self.aval._gt(self, other)

    def __lt__(self, other):
        return self.aval._lt(self, other)

    def __bool__(self):
        return self.aval._bool(self)

    def __nonzero__(self):
        return self.aval._nonzero(self)

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


def swap(f):
    return lambda x, y: f(y, x)


class ShapedArray:
    array_abstraction_level: int = 1
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    _neg = staticmethod(neg)
    _add = staticmethod(add)
    _radd = staticmethod(swap(add))
    _mul = staticmethod(mul)
    _rmul = staticmethod(swap(mul))
    _gt = staticmethod(greater)
    _lt = staticmethod(less)

    @staticmethod
    def _bool(tracer):
        raise Exception("ShapedArray can't be unambiguously converted to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("ShapedArray can't be unambiguously converted to bool")

    def str_short(self):
        return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __repr__(self):
        return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"


class ConcreteArray(ShapedArray):
    array_abstraction_level = 2
    val: np.ndarray

    def __init__(self, val):
        self.val = val
        self.shape = val.shape
        self.dtype = val.dtype

    @staticmethod
    def _bool(tracer: Tracer) -> bool:
        return bool(tracer.aval.val)

    @staticmethod
    def _nonzero(tracer: Tracer) -> bool:
        return bool(tracer.aval.val)


def get_aval(x):
    if isinstance(x, Tracer):
        return x.aval
    elif type(x) in jax_types:
        return ConcreteArray(np.asarray(x))
    else:
        raise TypeError(x)


jax_types = {
    bool,
    int,
    float,
    np.bool_,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.ndarray,
}
