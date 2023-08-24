__all__ = ["is_quantity", "grad", "value_and_grad"]

import jax
from jax._src.util import wraps
from jax.tree_util import tree_map


def is_quantity(obj):
    return hasattr(obj, "_units") and hasattr(obj, "_magnitude")


def grad(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    allow_int=False,
    reduce_axes=(),
):
    value_and_grad_f = value_and_grad(
        fun,
        argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    docstr = (
        "Gradient of {fun} with respect to positional argument(s) "
        "{argnums}. Takes the same arguments as {fun} but returns the "
        "gradient, which has the same shape as the arguments at "
        "positions {argnums}."
    )

    @wraps(fun, docstr=docstr, argnums=argnums)
    def grad_f(*args, **kwargs):
        _, g = value_and_grad_f(*args, **kwargs)
        return g

    @wraps(fun, docstr=docstr, argnums=argnums)
    def grad_f_aux(*args, **kwargs):
        (_, aux), g = value_and_grad_f(*args, **kwargs)
        return g, aux

    return grad_f_aux if has_aux else grad_f


def value_and_grad(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    allow_int=False,
    reduce_axes=(),
):
    # inspired by: https://twitter.com/shoyer/status/1531703890512490499
    docstr = (
        "Value and gradient of {fun} with respect to positional "
        "argument(s) {argnums}. Takes the same arguments as {fun} but "
        "returns a two-element tuple where the first element is the value "
        "of {fun} and the second element is the gradient, which has the "
        "same shape as the arguments at positions {argnums}."
    )

    def fun_wo_units(*args, **kwargs):
        if has_aux:
            result, aux = fun(*args, **kwargs)
        else:
            result = fun(*args, **kwargs)
        if is_quantity(result):
            magnitude = result.magnitude
            units = result.units
        else:
            magnitude = result
            units = None
        if has_aux:
            return magnitude, (units, aux)
        else:
            return magnitude, (units, None)

    value_and_grad_fun = jax.value_and_grad(
        fun_wo_units,
        argnums=argnums,
        has_aux=True,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    @wraps(fun, docstr=docstr, argnums=argnums)
    def wrapped(*args, **kwargs):
        (result_wo_units, (result_units, aux)), grad = value_and_grad_fun(
            *args, **kwargs
        )

        if result_units is None:
            result = result_wo_units
            grad = tree_map(
                lambda g: (g.magnitude / g.units if is_quantity(g) else g),
                grad,
                is_leaf=is_quantity,
            )

        else:
            result = result_wo_units * result_units
            grad = tree_map(
                lambda g: (
                    g.magnitude * result_units / g.units
                    if is_quantity(g)
                    else g * result_units
                ),
                grad,
                is_leaf=is_quantity,
            )

        if has_aux:
            return (result, aux), grad
        else:
            return result, grad

    return wrapped
