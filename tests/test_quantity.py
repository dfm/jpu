# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units

from jax_astropy_units import Quantity


def test_simple():
    x = jnp.array([1.0, 5.4, 0.1, -2.3])
    q = Quantity(x, units.km)
    np.testing.assert_allclose(q.to(units.m).value, 1000 * x)

    @jax.jit
    def func(arg):
        return arg.to(units.m).value

    np.testing.assert_allclose(func(q), 1000 * x)


def test_astropy_quantity():
    x = jnp.array([1.0, 5.4, 0.1, -2.3])
    q = x * units.km

    @jax.jit
    def func(arg):
        return arg.to(units.m)

    np.testing.assert_allclose(func(q).value, 1000 * x)


def test_init():
    x = jnp.array([1.0, 5.4, 0.1, -2.3])

    q = Quantity(x * units.m)
    assert q.unit is units.m
    np.testing.assert_allclose(q.value, x)

    q = Quantity(Quantity(x, units.m))
    assert q.unit is units.m
    np.testing.assert_allclose(q.value, x)

    q = Quantity(Quantity(x, units.m), units.cm)
    assert q.unit is units.cm
    np.testing.assert_allclose(q.value, x * 1e2)


@pytest.mark.parametrize(
    "a_,b_",
    [
        (jnp.array([1.0, 5.4, 0.1, -2.3]), jnp.array([-2.3, 5.4, 1.0, 0.1])),
        (jnp.array([1.0, 5.4, 0.1, -2.3]), 0.7),
    ],
)
def test_ops(a_, b_):
    for a, b in [
        (Quantity(a_, units.m), Quantity(b_, units.cm)),
        (a_ * units.m, Quantity(b_, units.cm)),
        (Quantity(a_, units.m), b_ * units.cm),
    ]:
        np.testing.assert_allclose((a + b).to(units.m).value, a_ + b_ * 1e-2)
        np.testing.assert_allclose((a - b).to(units.m).value, a_ - b_ * 1e-2)
        np.testing.assert_allclose(
            (a * b).to(units.m**2).value, a_ * b_ * 1e-2
        )
        np.testing.assert_allclose(
            (a / b).to(units.dimensionless_unscaled).value, a_ / (b_ * 1e-2)
        )

        if jnp.ndim(b_) > 0:
            np.testing.assert_allclose(
                (a @ b).to(units.m**2).value, a_ @ b_ * 1e-2
            )


# def test_numpy():
#     x = jnp.array([1.0, 5.4, 0.1, -2.3])
#     q = Quantity(x, units.km)
#     np.testing.assert_allclose(jnp.sin(q / units.km), jnp.sin(x))
