import jax
import jax.numpy as jnp
import pytest
from jax._src.public_test_util import check_close

import jpu.numpy as jnpu
from jpu import UnitRegistry, core

ureg = UnitRegistry()


@pytest.mark.parametrize(
    "params",
    [
        (
            lambda x: (jnpu.sin(2 * jnp.pi * x / (10.0 * ureg.m)) * ureg.s),
            (),
            ureg.m,
            ureg.s / ureg.m,
        ),
        (
            lambda x: (1.0 + x.magnitude),
            (),
            ureg.m,
            1 / ureg.m,
        ),
        (
            lambda x: jnp.sin(1.0 + x),
            (),
            None,
            None,
        ),
        (
            lambda x: jnpu.sum(jnpu.sin(2 * jnp.pi * x / (10.0 * ureg.m)) * ureg.s),
            (5, 2),
            ureg.m,
            ureg.s / ureg.m,
        ),
    ],
)
def test_grad(params):
    func, shape, in_units, grad_units = params

    def inp(x):
        return x if in_units is None else x * in_units

    def func_(x):
        y = func(inp(x))
        if hasattr(y, "magnitude"):
            return y.magnitude
        else:
            return y

    computed = core.grad(func)(inp(jnp.full(shape, 5.0)))
    expected = jax.grad(func_)(jnp.full(shape, 5.0))

    if grad_units is None:
        check_close(computed, expected)
        assert not hasattr(computed, "units")
    else:
        check_close(computed.magnitude, expected)  # type: ignore
        assert computed.units == grad_units  # type: ignore
