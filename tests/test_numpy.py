# mypy: ignore-errors

import operator

import jax
import jax.numpy as jnp
import numpy as np
import pint
import pytest
from jax._src.public_test_util import check_close

import jpu.numpy as jun
from jpu import UnitRegistry


def is_quantity(q):
    return hasattr(q, "magnitude") and hasattr(q, "units")


def assert_quantity_allclose(a, b):
    if is_quantity(a):
        assert is_quantity(b)
        assert str(a.units) == str(b.units)
        check_close(a.magnitude, b.magnitude)
    else:
        assert not is_quantity(b)
        check_close(a, b)


def test_type_wrapping():
    u = UnitRegistry()
    x = jnp.array([1.4, 2.0, -5.9])
    q = x * u.kpc
    assert q.units == u.kpc
    check_close(q.magnitude, x)
    assert type(q.magnitude) == type(x)


def test_array_ops():
    u = UnitRegistry()
    x = jnp.array([1.4, 2.0, -5.9])
    q = x * u.kpc

    # Addition
    res = q + np.array(0.01) * u.Mpc
    assert res.units == u.kpc
    check_close(res.magnitude, x + 10)
    assert type(res.magnitude) == type(x)

    # Different order
    res = np.array(0.01) * u.Mpc + q
    assert res.units == u.Mpc
    check_close(res.magnitude, 1e-3 * (x + 10))
    assert type(res.magnitude) == type(x)

    # Subtraction
    res = q - np.array(0.01) * u.Mpc
    assert res.units == u.kpc
    check_close(res.magnitude, x - 10)
    assert type(res.magnitude) == type(x)

    # Multiplication
    res = 2 * q
    assert res.units == u.kpc
    check_close(res.magnitude, 2 * x)
    assert type(res.magnitude) == type(x)

    # Division
    res = q / (2 * u.kpc)
    assert res.units == u.dimensionless
    check_close(res.magnitude, 0.5 * x)
    assert type(res.magnitude) == type(x)


@pytest.mark.parametrize(
    "func,in_unit",
    [
        ("exp", [""]),
        ("log", [""]),
        ("sin", ["degree"]),
        ("sin", ["radian"]),
        ("arctan2", ["m", "m"]),
        ("arctan2", ["m", "foot"]),
        ("argsort", ["day"]),
        ("std", ["day"]),
        ("var", ["m"]),
        ("dot", ["m", "s"]),
        ("median", ["m"]),
        ("cumprod", [""]),
        ("any", ["kpc"]),
    ],
)
def test_unary(func, in_unit):
    f = (lambda x: x**2) if func == "log" else (lambda x: x)

    pu = pint.UnitRegistry()
    np_args = []
    for n, iu in enumerate(in_unit):
        x = f(np.array([1.4, 2.0, -5.9]) - n) * pu(iu)
        np_args.append(x)

    u = UnitRegistry()
    jun_args = []
    for n, iu in enumerate(in_unit):
        x = f(jnp.array([1.4, 2.0, -5.9]) - n) * u(iu)
        jun_args.append(x)

    np_func = getattr(np, func)
    np_res = np_func(*np_args)

    jun_func = getattr(jun, func)
    jun_res = jun_func(*jun_args)
    assert_quantity_allclose(jun_res, np_res)
    jun_res = jax.jit(jun_func)(*jun_args)
    assert_quantity_allclose(jun_res, np_res)

    np_res_no_units = np_func(*(x.magnitude for x in np_args))
    jun_res_no_units = jun_func(*(x.magnitude for x in jun_args))
    check_close(jun_res_no_units, np_res_no_units)


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_duck_type(op):
    u = UnitRegistry()

    @jax.jit
    def func(x):
        return op(jnp.array(5.0), x)

    check_close(func(10.0 * u.dimensionless).magnitude, op(5.0, 10.0))
