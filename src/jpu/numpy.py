# mypy: ignore-errors

__all__ = [
    "modf",
    "frexp",
    "power",
    "add",
    "subtract",
    "meshgrid",
    "full_like",
    "interp",
    "where",
    "concatenate",
    "stack",
    "einsum",
    "any",
    "all",
]

import jax.numpy as jnp
from pint.numpy_func import (
    _get_first_input_units,
    _is_quantity,
    convert_to_consistent_units,
    get_op_output_unit,
    unwrap_and_wrap_consistent_units,
)


def modf(x, *args, **kwargs):
    (x,), output_wrap = unwrap_and_wrap_consistent_units(x)
    return tuple(output_wrap(y) for y in jnp.modf(x, *args, **kwargs))


def frexp(x, *args, **kwargs):
    (x,), output_wrap = unwrap_and_wrap_consistent_units(x)
    mantissa, exponent = jnp.frexp(x, *args, **kwargs)
    return output_wrap(mantissa), exponent


def power(x1, x2):
    if _is_quantity(x1):
        return x1**x2
    else:
        return x2.__rpow__(x1)


def add(x1, x2, *args, **kwargs):
    (x1, x2), output_wrap = unwrap_and_wrap_consistent_units(x1, x2)
    return output_wrap(jnp.add(x1, x2, *args, **kwargs))


def subtract(x1, x2, *args, **kwargs):
    (x1, x2), output_wrap = unwrap_and_wrap_consistent_units(x1, x2)
    return output_wrap(jnp.subtract(x1, x2, *args, **kwargs))


def meshgrid(*xi, **kwargs):
    # Simply need to map input units to onto list of outputs
    input_units = (x.units for x in xi)
    res = jnp.meshgrid(*(x.m for x in xi), **kwargs)
    return [out * unit for out, unit in zip(res, input_units)]


def full_like(a, fill_value, **kwargs):
    # Make full_like by multiplying with array from ones_like in a
    # non-multiplicative-unit-safe way
    if hasattr(fill_value, "_REGISTRY"):
        return fill_value._REGISTRY.Quantity(
            (jnp.ones_like(a, **kwargs) * fill_value.m),
            fill_value.units,
        )
    else:
        return jnp.ones_like(a, **kwargs) * fill_value


def interp(x, xp, fp, left=None, right=None, period=None):
    # Need to handle x and y units separately
    (x, xp, period), _ = unwrap_and_wrap_consistent_units(x, xp, period)
    (fp, right, left), output_wrap = unwrap_and_wrap_consistent_units(
        fp, left, right
    )
    return output_wrap(
        jnp.interp(x, xp, fp, left=left, right=right, period=period)
    )


def where(condition, *args):
    args, output_wrap = unwrap_and_wrap_consistent_units(*args)
    return output_wrap(jnp.where(condition, *args))


def concatenate(sequence, *args, **kwargs):
    sequence, output_wrap = unwrap_and_wrap_consistent_units(*sequence)
    return output_wrap(jnp.concatenate(sequence, *args, **kwargs))


def stack(arrays, *args, **kwargs):
    arrays, output_wrap = unwrap_and_wrap_consistent_units(*arrays)
    return output_wrap(jnp.stack(arrays, *args, **kwargs))


def einsum(subscripts, *operands, **kwargs):
    if not any(_is_quantity(x) for x in operands):
        return jnp.einsum(subscripts, *operands, **kwargs)
    operand_magnitudes, _ = convert_to_consistent_units(
        *operands, pre_calc_units=None
    )
    output_unit = get_op_output_unit(
        "mul", _get_first_input_units(operands), operands
    )
    return jnp.einsum(subscripts, *operand_magnitudes, **kwargs) * output_unit


def any(a, *args, **kwargs):
    if not _is_quantity(a):
        return jnp.any(a, *args, **kwargs)
    if a._is_multiplicative:
        return jnp.any(a._magnitude, *args, **kwargs)
    else:
        raise ValueError(
            "Boolean value of Quantity with offset unit is ambiguous."
        )


def all(a, *args, **kwargs):
    if not _is_quantity(a):
        return jnp.all(a, *args, **kwargs)
    if a._is_multiplicative:
        return jnp.all(a._magnitude, *args, **kwargs)
    else:
        raise ValueError(
            "Boolean value of Quantity with offset unit is ambiguous."
        )
