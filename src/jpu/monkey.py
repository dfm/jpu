__all__ = ["patch"]

import types
from functools import singledispatch

import jax.numpy as jnp

import jpu.numpy as jpunp
from jpu.registry import UnitRegistry


def patch():
    """
    Replace all the supported methods in jax.numpy with the unit-aware versions
    """
    funcs = _get_namespace_functions(jpunp)
    for name, func in funcs.items():
        if name.startswith("_"):
            continue
        jfunc = singledispatch(getattr(jnp, name))
        setattr(jnp, name, jfunc)
        jfunc.register(UnitRegistry.Quantity)(func)


def _get_namespace_functions(module):
    module_fns = {}
    for key in dir(module):
        if key in ("__getattr__", "__dir__"):
            continue
        try:
            attr = getattr(module, key)
        except Exception:
            continue
        if isinstance(
            attr,
            (
                types.BuiltinFunctionType,
                types.FunctionType,
                types.BuiltinMethodType,
                types.MethodType,
            ),
        ):
            module_fns[key] = attr
    return module_fns
