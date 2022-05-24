**A [JAX](https://jax.readthedocs.io/en/latest/) + [AstroPy
units](https://docs.astropy.org/en/stable/units/index.html) mashup.**

This is meant as a proof of concept to show how one might go about supporting
units as JAX arguments. It's non-trivial at this point to implement numpy
`ufunc`s for these custom types because of technical reasons (see
`__jax_array__` interface controversies, for example). Importantly the
`__jax_array__` and `pytree` interfaces are not currently compatible. So, until
a better custom dispatching interface is supported, the basic idea is to provide
a `pytree` type that knows about units, and then overload the `jax.numpy`
functions that we need.

Here's an example for the kind of way that you might use this proof of concept:

```python
import jax
import jax.numpy as jnp
import numpy as np
import astropy.units as u
import astropy.constants as c
from jax_astropy_units import Quantity

@jax.jit
def projectile_motion(v_init, theta, time, g=c.g0):
    """Compute the motion of a projectile with support for units"""

    v_init = Quantity(v_init, unit=u.m / u.s).value
    theta = Quantity(theta, unit=u.rad).value
    time = Quantity(time, unit=u.s).value
    g = Quantity(g, u.m / u.s**2).value

    x = v_init * time * jnp.cos(theta)
    y = v_init * time * jnp.sin(theta) - 0.5 * g * jnp.square(time)
    return Quantity(x, u.m), Quantity(y, u.m)

x, y = projectile_motion(
    0.01 * u.km / u.h, 15 * u.deg, np.linspace(0, 10, 50) * u.min
)
```

The key point is that this function can take parameters with any (compatible!)
units, while we can still be confident that the units will be correct when used
in the function body.
