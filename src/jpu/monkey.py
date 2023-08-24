__all__ = ["patch"]

import types
from functools import singledispatch

import jax.numpy as jnp
import numpy as np
import pint

import jpu.numpy as jpunp


def patch():
    """
    Replace all the supported methods in jax.numpy with the unit-aware versions
    """
    funcs = _get_module_functions(jpunp)
    for name, func in funcs.items():
        if name.startswith("_"):
            continue
        jfunc = singledispatch(getattr(jnp, name))
        setattr(jnp, name, jfunc)
        jfunc.register(pint.Quantity)(func)


def _get_module_functions(module):
    module_fns = {}
    for key in dir(module):
        # Omitting module level __getattr__, __dir__ which was added in Python 3.7
        # https://www.python.org/dev/peps/pep-0562/
        if key in ("__getattr__", "__dir__"):
            continue
        attr = getattr(module, key)
        if isinstance(attr, (types.BuiltinFunctionType, types.FunctionType, np.ufunc)):
            module_fns[key] = attr
    return module_fns
