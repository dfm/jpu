# mypy: ignore-errors

__all__ = ["with_units"]

from functools import wraps

import jax.linear_util as lu
from jax import core
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import ad, mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.tree_util import tree_flatten, tree_unflatten


def with_units(func, *, in_units, out_units):
    @wraps(func)
    def wrapped(*args, **kwargs):
        assert not kwargs

        args_flat, in_tree = tree_flatten(args)
        in_units_flat, _ = tree_flatten(in_units)

        flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(func), in_tree)
        out_units_flat, _ = tree_flatten(out_units)

        in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
        debug = pe.debug_info(func, in_tree, False, "with_units")
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
        assert not len(consts)
        closed_call = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
        out_flat = with_units_p.bind(
            *consts,
            *args_flat,
            call=closed_call,
            in_units=in_units_flat,
            out_units=out_units_flat,
        )
        return tree_unflatten(out_tree(), out_flat)

    return wrapped


def _with_units_impl(*args, call: core.ClosedJaxpr, in_units, out_units):
    del in_units, out_units
    return core.jaxpr_as_fun(call)(*args)


def _with_units_abstract_eval(*in_avals, call: core.ClosedJaxpr, **_):
    return call.out_avals


def _with_units_jvp(primals, tangents, *, call, in_units, out_units):
    tangents = map(ad.instantiate_zeros, tangents)
    jvp_call, _ = ad.jvp_jaxpr(call, [True] * len(primals), True)
    outs = with_units_p.bind(
        *primals,
        *tangents,
        call=jvp_call,
        in_units=tuple(in_units) + tuple(in_units),
        out_units=tuple(out_units) + tuple(out_units),  # FIXME
    )
    assert len(outs) % 2 == 0, len(outs)
    return outs[: len(outs) // 2], outs[len(outs) // 2 :]


with_units_p = core.Primitive("with_units")
with_units_p.multiple_results = True
with_units_p.def_impl(_with_units_impl)
with_units_p.def_abstract_eval(_with_units_abstract_eval)
ad.primitive_jvps[with_units_p] = _with_units_jvp

xla.register_initial_style_primitive(with_units_p)
mlir.register_lowering(
    with_units_p, mlir.lower_fun(_with_units_impl, multiple_results=True)
)
