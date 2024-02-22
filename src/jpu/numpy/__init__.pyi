from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Literal, TypeVar, overload

import numpy as _np
from jax._src.lax.lax import PrecisionLike
from jax._src.typing import DimSize, DType, DTypeLike, DuckTypedArray, Shape

from jpu.registry import Quantity

ArrayLike = Quantity
Array = Quantity

_T = TypeVar("_T")
_Axis = None | int | Sequence[int]

def abs(x: ArrayLike, /) -> Array: ...
def absolute(x: ArrayLike, /) -> Array: ...
def add(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def amax(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def amin(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def all(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = ...,
    atol: ArrayLike = ...,
    equal_nan: bool = ...,
) -> Array: ...
def any(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def append(arr: ArrayLike, values: ArrayLike, axis: int | None = ...) -> Array: ...
def arange(
    start: DimSize,
    stop: DimSize | None = ...,
    step: DimSize | None = ...,
    dtype: DTypeLike | None = ...,
) -> Array: ...
def arccos(x: ArrayLike, /) -> Array: ...
def arccosh(x: ArrayLike, /) -> Array: ...
def arcsin(x: ArrayLike, /) -> Array: ...
def arcsinh(x: ArrayLike, /) -> Array: ...
def arctan(x: ArrayLike, /) -> Array: ...
def arctan2(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def arctanh(x: ArrayLike, /) -> Array: ...
def argmax(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: bool | None = ...,
) -> Array: ...
def argmin(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: bool | None = ...,
) -> Array: ...
def argpartition(a: ArrayLike, kth: int, axis: int = ...) -> Array: ...
def argsort(
    a: ArrayLike,
    axis: int | None = ...,
    kind: str | None = ...,
    order: None = ...,
    *,
    stable: bool = ...,
    descending: bool = ...,
) -> Array: ...

around = round

def asin(x: ArrayLike, /) -> Array: ...
def asinh(x: ArrayLike, /) -> Array: ...
def astype(a: ArrayLike, dtype: DTypeLike | None, /, *, copy: bool = ...) -> Array: ...
def atan(x: ArrayLike, /) -> Array: ...
def atan2(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def atanh(x: ArrayLike, /) -> Array: ...
@overload
def atleast_1d() -> list[Array]: ...
@overload
def atleast_1d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_1d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def atleast_2d() -> list[Array]: ...
@overload
def atleast_2d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_2d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def atleast_3d() -> list[Array]: ...
@overload
def atleast_3d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_3d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def average(
    a: ArrayLike,
    axis: _Axis = ...,
    weights: ArrayLike | None = ...,
    returned: Literal[False] = False,
    keepdims: bool = False,
) -> Array: ...
@overload
def average(
    a: ArrayLike,
    axis: _Axis = ...,
    weights: ArrayLike | None = ...,
    *,
    returned: Literal[True],
    keepdims: bool = False,
) -> tuple[Array, Array]: ...
@overload
def average(
    a: ArrayLike,
    axis: _Axis = ...,
    weights: ArrayLike | None = ...,
    returned: bool = False,
    keepdims: bool = False,
) -> Array | tuple[Array, Array]: ...
def block(
    arrays: ArrayLike | Sequence[ArrayLike] | Sequence[Sequence[ArrayLike]],
) -> Array: ...
def broadcast_arrays(*args: ArrayLike) -> list[Array]: ...
def broadcast_to(array: ArrayLike, shape: DimSize | Shape) -> Array: ...
def cbrt(x: ArrayLike, /) -> Array: ...
def ceil(x: ArrayLike, /) -> Array: ...
def choose(
    a: ArrayLike, choices: Sequence[ArrayLike], out: None = ..., mode: str = ...
) -> Array: ...
def clip(
    a: ArrayLike,
    a_min: ArrayLike | None = ...,
    a_max: ArrayLike | None = ...,
    out: None = ...,
) -> Array: ...
def column_stack(tup: _np.ndarray | Array | Sequence[ArrayLike]) -> Array: ...
def compress(
    condition: ArrayLike, a: ArrayLike, axis: int | None = ..., out: None = ...
) -> Array: ...
def concat(arrays: Sequence[ArrayLike], /, *, axis: int | None = 0) -> Array: ...
def concatenate(
    arrays: _np.ndarray | Array | Sequence[ArrayLike],
    axis: int | None = ...,
    dtype: DTypeLike | None = ...,
) -> Array: ...
def conjugate(x: ArrayLike, /) -> Array: ...

conj = conjugate

def convolve(
    a: ArrayLike,
    v: ArrayLike,
    mode: str = ...,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DType | None = ...,
) -> Array: ...
def copy(a: ArrayLike, order: str | None = ...) -> Array: ...
def copysign(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def corrcoef(x: ArrayLike, y: ArrayLike | None = ..., rowvar: bool = ...) -> Array: ...
def correlate(
    a: ArrayLike,
    v: ArrayLike,
    mode: str = ...,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DType | None = ...,
) -> Array: ...
def cos(x: ArrayLike, /) -> Array: ...
def cosh(x: ArrayLike, /) -> Array: ...
def count_nonzero(a: ArrayLike, axis: _Axis = ..., keepdims: bool = ...) -> Array: ...
def cov(
    m: ArrayLike,
    y: ArrayLike | None = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: int | None = ...,
    fweights: ArrayLike | None = ...,
    aweights: ArrayLike | None = ...,
) -> Array: ...
def cross(
    a: ArrayLike,
    b: ArrayLike,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = ...,
) -> Array: ...
def cumprod(
    a: ArrayLike, axis: _Axis = ..., dtype: DTypeLike = ..., out: None = ...
) -> Array: ...
def cumsum(
    a: ArrayLike, axis: _Axis = ..., dtype: DTypeLike = ..., out: None = ...
) -> Array: ...
def deg2rad(x: ArrayLike, /) -> Array: ...
def delete(
    arr: ArrayLike,
    obj: ArrayLike | slice,
    axis: int | None = ...,
    *,
    assume_unique_indices: bool = ...,
) -> Array: ...
def diag(v: ArrayLike, k: int = 0) -> Array: ...
def diag_indices(n: int, ndim: int = ...) -> tuple[Array, ...]: ...
def diag_indices_from(arr: ArrayLike) -> tuple[Array, ...]: ...
def diagflat(v: ArrayLike, k: int = 0) -> Array: ...
def diagonal(
    a: ArrayLike, offset: ArrayLike = ..., axis1: int = ..., axis2: int = ...
): ...
def diff(
    a: ArrayLike,
    n: int = ...,
    axis: int = ...,
    prepend: ArrayLike | None = ...,
    append: ArrayLike | None = ...,
) -> Array: ...
def digitize(x: ArrayLike, bins: ArrayLike, right: bool = ...) -> Array: ...
def divmod(x: ArrayLike, y: ArrayLike, /) -> tuple[Array, Array]: ...
def dot(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
def dsplit(ary: ArrayLike, indices_or_sections: int | ArrayLike) -> list[Array]: ...
def dstack(
    tup: _np.ndarray | Array | Sequence[ArrayLike],
    dtype: DTypeLike | None = ...,
) -> Array: ...
def ediff1d(
    ary: ArrayLike,
    to_end: ArrayLike | None = ...,
    to_begin: ArrayLike | None = ...,
) -> Array: ...
@overload
def einsum(
    subscript: str,
    /,
    *operands: ArrayLike,
    out: None = ...,
    optimize: str = "optimal",
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
    _use_xeinsum: bool = False,
    _dot_general: Callable[..., Array] = ...,
) -> Array: ...
@overload
def einsum(
    arr: ArrayLike,
    axes: Sequence[Any],
    /,
    *operands: ArrayLike | Sequence[Any],
    out: None = ...,
    optimize: str = "optimal",
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
    _use_xeinsum: bool = False,
    _dot_general: Callable[..., Array] = ...,
) -> Array: ...
@overload
def einsum(
    subscripts,
    /,
    *operands,
    out: None = ...,
    optimize: str = ...,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
    _use_xeinsum: bool = ...,
    _dot_general: Callable[..., Array] = ...,
) -> Array: ...
def einsum_path(subscripts, *operands, optimize=...): ...
def empty(shape: Any, dtype: DTypeLike | None = ...) -> Array: ...
def empty_like(
    prototype: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
) -> Array: ...
def equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def exp(x: ArrayLike, /) -> Array: ...
def exp2(x: ArrayLike, /) -> Array: ...
def expand_dims(a: ArrayLike, axis: int | Sequence[int]) -> Array: ...
def expm1(x: ArrayLike, /) -> Array: ...
def extract(condition: ArrayLike, arr: ArrayLike) -> Array: ...
def eye(
    N: DimSize,
    M: DimSize | None = ...,
    k: int = ...,
    dtype: DTypeLike | None = ...,
) -> Array: ...
def fabs(x: ArrayLike, /) -> Array: ...
def fix(x: ArrayLike, out: None = ...) -> Array: ...
def flatnonzero(
    a: ArrayLike,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike] = ...,
) -> Array: ...
def flip(m: ArrayLike, axis: int | Sequence[int] | None = ...) -> Array: ...
def floor(x: ArrayLike, /) -> Array: ...
def floor_divide(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def fmax(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def fmin(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def fmod(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def frexp(x: ArrayLike, /) -> tuple[Array, Array]: ...
def full(shape: Any, fill_value: ArrayLike, dtype: DTypeLike | None = ...) -> Array: ...
def full_like(
    a: ArrayLike | DuckTypedArray,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
) -> Array: ...
def gradient(
    f: ArrayLike,
    *varargs: ArrayLike,
    axis: int | Sequence[int] | None = ...,
    edge_order: int | None = ...,
) -> Array | list[Array]: ...
def greater(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def greater_equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def hstack(
    tup: _np.ndarray | Array | Sequence[ArrayLike],
    dtype: DTypeLike | None = ...,
) -> Array: ...
def hypot(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def identity(n: DimSize, dtype: DTypeLike | None = ...) -> Array: ...
def imag(x: ArrayLike, /) -> Array: ...
def inner(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
def insert(
    arr: ArrayLike,
    obj: ArrayLike | slice,
    values: ArrayLike,
    axis: int | None = ...,
) -> Array: ...
def interp(
    x: ArrayLike,
    xp: ArrayLike,
    fp: ArrayLike,
    left: ArrayLike | str | None = ...,
    right: ArrayLike | str | None = ...,
    period: ArrayLike | None = ...,
) -> Array: ...
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = ...,
    return_indices: bool = ...,
) -> Array | tuple[Array, Array, Array]: ...
def invert(x: ArrayLike, /) -> Array: ...
def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = ...,
    atol: ArrayLike = ...,
    equal_nan: bool = ...,
) -> Array: ...
def iscomplex(m: ArrayLike) -> Array: ...
def isfinite(x: ArrayLike, /) -> Array: ...
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = ...,
    invert: bool = ...,
) -> Array: ...
def isinf(x: ArrayLike, /) -> Array: ...
def isnan(x: ArrayLike, /) -> Array: ...
def isreal(m: ArrayLike) -> Array: ...
def ldexp(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def less(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def less_equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = ...,
    axis: int = 0,
) -> Array: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int,
    endpoint: bool,
    retstep: Literal[True],
    dtype: DTypeLike | None = ...,
    axis: int = 0,
) -> tuple[Array, Array]: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    *,
    retstep: Literal[True],
    dtype: DTypeLike | None = ...,
    axis: int = 0,
) -> tuple[Array, Array]: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = ...,
    axis: int = 0,
) -> Array | tuple[Array, Array]: ...
def log(x: ArrayLike, /) -> Array: ...
def log10(x: ArrayLike, /) -> Array: ...
def log1p(x: ArrayLike, /) -> Array: ...
def log2(x: ArrayLike, /) -> Array: ...
def logaddexp(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def logaddexp2(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def matmul(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...

max = amax

def maximum(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def mean(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def median(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> Array: ...

min = amin

def minimum(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def mod(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def modf(x: ArrayLike, /, out=None) -> tuple[Array, Array]: ...
def moveaxis(
    a: ArrayLike,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Array: ...
def multiply(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def nan_to_num(
    x: ArrayLike,
    copy: bool = ...,
    nan: ArrayLike = ...,
    posinf: ArrayLike | None = ...,
    neginf: ArrayLike | None = ...,
) -> Array: ...
def nanargmax(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: bool | None = ...,
) -> Array: ...
def nanargmin(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: bool | None = ...,
) -> Array: ...
def nancumprod(
    a: ArrayLike, axis: _Axis = ..., dtype: DTypeLike = ..., out: None = ...
) -> Array: ...
def nancumsum(
    a: ArrayLike, axis: _Axis = ..., dtype: DTypeLike = ..., out: None = ...
) -> Array: ...
def nanmax(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanmean(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanmedian(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> Array: ...
def nanmin(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanpercentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: str = ...,
    keepdims: bool = ...,
    interpolation: None = ...,
) -> Array: ...
def nanprod(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanquantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: str = ...,
    keepdims: bool = ...,
    interpolation: None = ...,
) -> Array: ...
def nanstd(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: bool = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nansum(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanvar(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    ddof: int = 0,
    keepdims: bool = False,
    where: ArrayLike | None = ...,
) -> Array: ...

ndim = _np.ndim

def negative(x: ArrayLike, /) -> Array: ...
def nextafter(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def nonzero(
    a: ArrayLike,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> tuple[Array, ...]: ...
def not_equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def ones(shape: Any, dtype: DTypeLike | None = ...) -> Array: ...
def ones_like(
    a: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
) -> Array: ...
def outer(a: ArrayLike, b: Array, out: None = ...) -> Array: ...

PadValueLike = _T | Sequence[_T] | Sequence[Sequence[_T]]

def pad(
    array: ArrayLike,
    pad_width: PadValueLike[int | Array | _np.ndarray],
    mode: str | Callable[..., Any] = ...,
    **kwargs,
) -> Array: ...
def partition(a: ArrayLike, kth: int, axis: int = ...) -> Array: ...
def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: str = ...,
    keepdims: bool = ...,
    interpolation: None = ...,
) -> Array: ...
def positive(x: ArrayLike, /) -> Array: ...
def pow(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def power(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def prod(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
    promote_integers: bool = ...,
) -> Array: ...
def ptp(
    a: ArrayLike, axis: _Axis = ..., out: None = ..., keepdims: bool = ...
) -> Array: ...
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: str = ...,
    keepdims: bool = ...,
    interpolation: None = ...,
) -> Array: ...
def rad2deg(x: ArrayLike, /) -> Array: ...
def ravel(a: ArrayLike, order: str = ...) -> Array: ...
def real(x: ArrayLike, /) -> Array: ...
def reciprocal(x: ArrayLike, /) -> Array: ...
def remainder(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def repeat(
    a: ArrayLike,
    repeats: ArrayLike,
    axis: int | None = ...,
    *,
    total_repeat_length: int | None = ...,
) -> Array: ...
def reshape(a: ArrayLike, newshape: DimSize | Shape, order: str = ...) -> Array: ...
def resize(a: ArrayLike, new_shape: Shape) -> Array: ...
def result_type(*args: Any) -> DType: ...
def rint(x: ArrayLike, /) -> Array: ...
def roll(
    a: ArrayLike,
    shift: ArrayLike | Sequence[int],
    axis: int | Sequence[int] | None = ...,
) -> Array: ...
def rollaxis(a: ArrayLike, axis: int, start: int = 0) -> Array: ...
def rot90(m: ArrayLike, k: int = ..., axes: tuple[int, int] = ...) -> Array: ...
def round(a: ArrayLike, decimals: int = ..., out: None = ...) -> Array: ...

round_ = round

def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: str = ...,
    sorter: None = ...,
    *,
    method: str = ...,
) -> Array: ...

shape = _np.shape

def sign(x: ArrayLike, /) -> Array: ...
def signbit(x: ArrayLike, /) -> Array: ...
def sin(x: ArrayLike, /) -> Array: ...
def sinh(x: ArrayLike, /) -> Array: ...

size = _np.size

def sort(
    a: ArrayLike,
    axis: int | None = ...,
    kind: str | None = ...,
    order: None = ...,
    *,
    stable: bool = ...,
    descending: bool = ...,
) -> Array: ...
def sqrt(x: ArrayLike, /) -> Array: ...
def square(x: ArrayLike, /) -> Array: ...
def squeeze(a: ArrayLike, axis: int | Sequence[int] | None = ...) -> Array: ...
def stack(
    arrays: _np.ndarray | Array | Sequence[ArrayLike],
    axis: int = ...,
    out: None = ...,
    dtype: DTypeLike | None = ...,
) -> Array: ...
def std(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def subtract(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def sum(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
    promote_integers: bool = ...,
) -> Array: ...
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> Array: ...
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    mode: str | None = ...,
    unique_indices: bool = ...,
    indices_are_sorted: bool = ...,
    fill_value: ArrayLike | None = ...,
) -> Array: ...
def tan(x: ArrayLike, /) -> Array: ...
def tanh(x: ArrayLike, /) -> Array: ...
def tile(A: ArrayLike, reps: DimSize | Sequence[DimSize]) -> Array: ...
def trace(
    a: ArrayLike,
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
) -> Array: ...
def transpose(a: ArrayLike, axes: Sequence[int] | None = ...) -> Array: ...
def trim_zeros(filt: ArrayLike, trim: str = ...) -> Array: ...
def true_divide(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def trunc(x: ArrayLike, /) -> Array: ...
def unwrap(
    p: ArrayLike,
    discont: ArrayLike | None = ...,
    axis: int = ...,
    period: ArrayLike = ...,
) -> Array: ...
def var(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def vstack(
    tup: _np.ndarray | Array | Sequence[ArrayLike],
    dtype: DTypeLike | None = ...,
) -> Array: ...
@overload
def where(
    condition: ArrayLike,
    x: Literal[None] = ...,
    y: Literal[None] = ...,
    /,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> tuple[Array, ...]: ...
@overload
def where(
    condition: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    /,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> Array: ...
@overload
def where(
    condition: ArrayLike,
    x: ArrayLike | None = ...,
    y: ArrayLike | None = ...,
    /,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> Array | tuple[Array, ...]: ...
def zeros(shape: Any, dtype: DTypeLike | None = ...) -> Array: ...
def zeros_like(
    a: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
) -> Array: ...

cumproduct = cumprod
degrees = rad2deg
divide = true_divide
radians = deg2rad
