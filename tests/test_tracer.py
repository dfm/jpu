# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np

from jpu.registry import UnitRegistry
from jpu.tracer import units


def test_decorator():
    @units
    def func(x, y):
        return jnp.exp(x / (0.5 * y) + 2.3)

    u = UnitRegistry()
    res = func(jnp.ones(3) * u.m, jnp.array([5.6]) * u.km)
    assert res.units == u.dimensionless
