import operator
import warnings
from functools import partial
from typing import Generic

import jax
import jax.numpy as jnp
from pint.facets.plain import MagnitudeT, PlainQuantity

from jpu import numpy as jpu_numpy

SUPPORTED_NUMPY_METHODS = [
    "all",
    "any",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "choose",
    "clip",
    "compress",
    "conj",
    "conjugate",
    "copy",
    "cumprod",
    "cumsum",
    "delete",
    "diagonal",
    "dot",
    "max",
    "mean",
    "min",
    "nonzero",
    "prod",
    "ptp",
    "ravel",
    "repeat",
    "reshape",
    "round",
    "searchsorted",
    "sort",
    "squeeze",
    "std",
    "sum",
    "swapaxes",
    "take",
    "trace",
    "transpose",
    "var",
]
SUPPORTED_PASSTHROUGH_METHODS = [
    "astype",
    "block_until_ready",
    "clone",
    "flatten",
    "item",
    "view",
]


class JpuQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
    def __array__(self, *args, **kwargs):
        warnings.warn(
            "The unit of a Quantity is stripped when downcasted to an array.",
            stacklevel=2,
        )
        return self._magnitude.__array__(*args, **kwargs)

    @property
    def dtype(self):
        return jnp.asarray(self._magnitude).dtype

    @property
    def ndim(self):
        return jnp.ndim(self._magnitude)

    @property
    def shape(self):
        return jnp.shape(self._magnitude)

    def _maybe_dimensionless(self, other):
        if isinstance(other, jax.Array):
            return self._REGISTRY.Quantity(other, self._REGISTRY.dimensionless)
        return other

    def __iadd__(self, other):
        return self._add_sub(self._maybe_dimensionless(other), operator.add)

    def __add__(self, other):
        return self._add_sub(self._maybe_dimensionless(other), operator.add)

    __radd__ = __add__

    def __isub__(self, other):
        return self._add_sub(self._maybe_dimensionless(other), operator.sub)

    def __sub__(self, other):
        return self._add_sub(self._maybe_dimensionless(other), operator.sub)

    def __rsub__(self, other):
        return -self._add_sub(self._maybe_dimensionless(other), operator.sub)

    def __len__(self):
        return len(self._magnitude)

    def _wrap_passthrough_method(self, name, *args, **kwargs):
        return self.__class__(
            getattr(self._magnitude, name)(*args, **kwargs), self._units
        )

    def __getitem__(self, key):
        return self.__class__(self._magnitude[key], self._units)

    def __getattr__(self, item):
        if item in SUPPORTED_NUMPY_METHODS:
            return partial(getattr(jpu_numpy, item), self)
        elif item in SUPPORTED_PASSTHROUGH_METHODS:
            return partial(self._wrap_passthrough_method, item)
        try:
            return getattr(self._magnitude, item)
        except AttributeError:
            raise AttributeError(
                f"Neither Quantity object nor its magnitude ({self._magnitude}) "
                f"has attribute '{item}'"
            ) from None
