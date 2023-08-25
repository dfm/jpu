# mypy: ignore-errors

import jax
import jax.numpy as jnp
from jax._src.public_test_util import check_close

from jpu.registry import UnitRegistry


def test_tree_flatten():
    u = UnitRegistry()
    x = jnp.array([1.4, 2.0, -5.9])
    q = x * u.m

    val, _ = jax.tree_util.tree_flatten(q)
    assert len(val) == 1
    check_close(val[0], x)


def test_jittable():
    u = UnitRegistry()
    x = jnp.array([1.4, 2.0, -5.9])
    q = x * u.m

    @jax.jit
    def func(q):
        assert q.u == u.m
        return q + 4.5 * u.km

    res = func(q)
    assert res.units == u.m
    check_close(res.magnitude, x + 4500.0)


def test_ducktype():
    u = UnitRegistry()
    x = jnp.array([1.4, 2.0, -5.9])
    q = x * u.m

    res = q.sum()
    assert res.units == u.m
    check_close(res.magnitude, x.sum())

    @jax.jit
    def func(q):
        print(q)
        return q.sum()

    res = func(q)
    print(type(q), type(res))
    assert res.units == u.m
    check_close(res.magnitude, x.sum())


def test_unary_ops():
    u = UnitRegistry()

    x = jnp.array([1.4, 2.0, -5.9])
    q = x * u.m

    for func in [
        lambda q: q**2,
        lambda q: q.sum(),
        lambda q: 2 * q,
    ]:
        res = func(q)
        check_close(res.magnitude, func(x))
        res = jax.jit(func)(q)
        check_close(res.magnitude, func(x))
