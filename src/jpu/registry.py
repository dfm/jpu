__all__ = ["UnitRegistry"]

from typing import Any

import jax
from jax.tree_util import register_pytree_node
from pint import UnitRegistry as PintUnitRegistry, compat


class UnitRegistry(PintUnitRegistry):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Register the Quantity produced by this registry with JAX
        def flatten_quantity(q: Any) -> tuple[tuple[Any], Any]:
            return (q.magnitude,), q.units

        def unflatten_quantity(aux_data: Any, children: tuple[Any]) -> Any:
            return self.Quantity(children[0], aux_data)

        register_pytree_node(self.Quantity, flatten_quantity, unflatten_quantity)


def is_duck_array_type(cls: Any) -> bool:
    return issubclass(cls, jax.core.Tracer) or _is_duck_array_type(cls)


_is_duck_array_type = compat.is_duck_array_type
compat.is_duck_array_type = is_duck_array_type
