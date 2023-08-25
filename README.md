# JAX + Units

**Built with [JAX](https://jax.readthedocs.io) and
[Pint](https://pint.readthedocs.io)!**

This module provides an interface between [JAX](https://jax.readthedocs.io) and
[Pint](https://pint.readthedocs.io) to allow JAX to support operations with
units. The propagation of units happens at trace time, so jitted functions
should see no runtime cost. This library is experimental so expect some sharp
edges.

For example:

```python
>>> import jax
>>> import jax.numpy as jnp
>>> import jpu
>>>
>>> u = jpu.UnitRegistry()
>>>
>>> @jax.jit
... def add_two_lengths(a, b):
...     return a + b
...
>>> add_two_lengths(3 * u.m, jnp.array([4.5, 1.2, 3.9]) * u.cm)
<Quantity([3.045 3.012 3.039], 'meter')>

```

## Installation

To install, use `pip`:

```bash
python -m pip install jpu
```

The only dependencies are `jax` and `pint`, and these will also be installed, if
not already in your environment. Take a look at [the JAX docs for more
information about installing JAX on different
systems](https://github.com/google/jax#installation).

## Usage

Here is a slightly more complete example:

```python
>>> import jax
>>> import numpy as np
>>> from jpu import UnitRegistry, numpy as jnpu
>>>
>>> u = UnitRegistry()
>>>
>>> @jax.jit
... def projectile_motion(v_init, theta, time, g=u.standard_gravity):
...     """Compute the motion of a projectile with support for units"""
...     x = v_init * time * jnpu.cos(theta)
...     y = v_init * time * jnpu.sin(theta) - 0.5 * g * jnpu.square(time)
...     return x.to(u.m), y.to(u.m)
...
>>> x, y = projectile_motion(
...     5.0 * u.km / u.h, 60 * u.deg, np.linspace(0, 1, 50) * u.s
... )

```

## Technical details & limitations

The most significant limitation of this library is the fact that users must use
`jpu.numpy` functions when interacting with "quantities" with units instead of
the `jax.numpy` interface. This is because JAX does not (yet?) provide a general
interface for dispatching of ufuncs on custom array classes. I have played
around with the undocumented `__jax_array__` interface, but it's not really
flexible enough, and it isn't currently compatible with Pytree objects.

So far, only a subset of the `numpy`/`jax.numpy` interface is implemented. Pull
requests adding broader support (including submodules) would be welcome!
