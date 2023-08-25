from typing import Any, Generic

from jax.tree_util import register_pytree_node
from pint import facets
from pint.compat import TypeAlias
from pint.facets.plain import GenericPlainRegistry, PlainUnit, QuantityT, UnitT

from jpu.quantity import JpuQuantity


class GenericJpuRegistry(
    Generic[QuantityT, UnitT], GenericPlainRegistry[QuantityT, UnitT]
):
    pass


class JpuRegistry(GenericPlainRegistry[JpuQuantity[Any], PlainUnit]):
    Quantity: TypeAlias = JpuQuantity[Any]  # type: ignore
    Unit: TypeAlias = PlainUnit


class GenericUnitRegistry(  # type: ignore
    Generic[facets.QuantityT, facets.UnitT],
    facets.GenericSystemRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericContextRegistry[facets.QuantityT, facets.UnitT],
    GenericJpuRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericNumpyRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericMeasurementRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericFormattingRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericNonMultiplicativeRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericPlainRegistry[facets.QuantityT, facets.UnitT],
):
    pass


class Quantity(
    facets.SystemRegistry.Quantity,
    facets.ContextRegistry.Quantity,
    JpuRegistry.Quantity,
    facets.NumpyRegistry.Quantity,
    facets.MeasurementRegistry.Quantity,
    facets.FormattingRegistry.Quantity,
    facets.NonMultiplicativeRegistry.Quantity,
    facets.PlainRegistry.Quantity,
):
    pass


class Unit(
    facets.SystemRegistry.Unit,
    facets.ContextRegistry.Unit,
    facets.NumpyRegistry.Unit,
    facets.MeasurementRegistry.Unit,
    facets.FormattingRegistry.Unit,
    facets.NonMultiplicativeRegistry.Unit,
    facets.PlainRegistry.Unit,
):
    pass


class UnitRegistry(GenericUnitRegistry[Quantity, Unit]):
    Quantity: TypeAlias = Quantity  # type: ignore
    Unit: TypeAlias = Unit

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def flatten_quantity(q):
            return (q.magnitude,), (q.units, q._REGISTRY)

        def unflatten_quantity(aux_data, children):
            (magnitude,) = children
            units, registry = aux_data
            return registry.Quantity(magnitude, units)

        register_pytree_node(self.Quantity, flatten_quantity, unflatten_quantity)
