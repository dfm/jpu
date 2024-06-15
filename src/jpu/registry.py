import pint
from jax.tree_util import register_pytree_node
from pint.compat import TypeAlias

from jpu.quantity import JpuQuantity


class UnitRegistry(pint.registry.GenericUnitRegistry[JpuQuantity, pint.Unit]):
    Quantity: TypeAlias = JpuQuantity
    Unit: TypeAlias = pint.Unit

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def flatten_quantity(q):
            return (q.magnitude,), (q.units, q._REGISTRY)

        def unflatten_quantity(aux_data, children):
            (magnitude,) = children
            units, registry = aux_data
            return registry.Quantity(magnitude, units)

        register_pytree_node(self.Quantity, flatten_quantity, unflatten_quantity)
