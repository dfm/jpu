# mypy: ignore-errors

__all__ = ["units"]

from functools import wraps
from typing import Any, Dict

import jax
from jax._src.util import safe_map

from . import numpy as jnu
from .registry import UnitRegistry

unit_transform_registry = {}
unit_transform_registry[jax.lax.mul_p] = lambda a, b: a * b
unit_transform_registry[jax.lax.div_p] = lambda a, b: a / b
unit_transform_registry[jax.lax.add_p] = lambda a, b: a + b
unit_transform_registry[jax.lax.exp_p] = jnu.exp


def units_jaxpr(jaxpr, literals, *args, input_units):
    unit_registry = None
    for a in args:
        unit_registry = getattr(a, "_REGISTRY", None)
    if unit_registry is None:
        unit_registry = UnitRegistry()

    env: Dict[Any, Any] = {}

    def read(var: Any) -> Any:
        if type(var) is jax.core.Literal:
            return var.val
        assert unit_registry is not None
        return unit_registry.Quantity(*env[var])

    def write(var: Any, val: Any, units: Any = None) -> None:
        if hasattr(val, "magnitude") and hasattr(val, "units"):
            if units is not None:
                val = val.to(units)
            env[var] = (val.magnitude, val.units)
        else:
            env[var] = (val, units)

    safe_map(write, jaxpr.invars, args, input_units)
    safe_map(write, jaxpr.constvars, literals)

    for eqn in jaxpr.eqns:
        if eqn.primitive not in unit_transform_registry:
            raise NotImplementedError(
                f"Primitive '{eqn.primitive}'' does not have registered unit transformation."
            )
        invals = safe_map(read, eqn.invars)
        outvals = unit_transform_registry[eqn.primitive](*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        safe_map(write, eqn.outvars, outvals)
    result = safe_map(read, jaxpr.outvars)
    if not eqn.primitive.multiple_results:
        return result[0]
    return result


def units(func, *, input_units=None):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if input_units is None:
            input_units_val = [None] * len(args)
        else:
            input_units_val = input_units
        args_no_units = [getattr(a, "magnitude", a) for a in args]
        closed_jaxpr = jax.make_jaxpr(func)(*args_no_units, **kwargs)
        result = units_jaxpr(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            *args,
            input_units=input_units_val,
        )
        return result

    return wrapped
