# mypy: ignore-errors


def test_readme():
    import jax
    import numpy as np

    from jpu import UnitRegistry, numpy as jnpu

    u = UnitRegistry()

    @jax.jit
    def projectile_motion(v_init, theta, time, g=u.standard_gravity):
        """Compute the motion of a projectile with support for units"""
        x = v_init * time * jnpu.cos(theta)
        y = v_init * time * jnpu.sin(theta) - 0.5 * g * jnpu.square(time)
        return x.to(u.m), y.to(u.m)

    x, y = projectile_motion(5.0 * u.km / u.h, 60 * u.deg, np.linspace(0, 1, 50) * u.s)
    assert x.units == u.m
    assert y.units == u.m
