__all__ = ["Quantity"]

import operator
from typing import Any, Callable, Iterable, Optional, Tuple

import jax.numpy as jnp
from astropy import units
from jax.tree_util import register_pytree_node_class


def _impl_additive_binary(
    op: Callable[[Any, Any], Any], reflect: bool = False
) -> Callable[[Any, Any], Any]:
    _op = (lambda a, b: op(b, a)) if reflect else op

    def func(self: "Quantity", other: Any) -> "Quantity":
        try:
            converted = other.to(self.unit)
        except AttributeError:
            return Quantity(_op(self.value, other), self.unit)
        return Quantity(_op(self.value, converted.value), self.unit)

    return func


def _impl_multiplicative_binary(
    op: Callable[[Any, Any], Any],
    unit_op: Optional[Callable[[Any, Any], Any]] = None,
    reflect: bool = False,
) -> Callable[[Any, Any], Any]:
    _op = (lambda a, b: op(b, a)) if reflect else op
    if unit_op is not None:
        _unit_op = (lambda a, b: unit_op(b, a)) if reflect else unit_op
    else:
        _unit_op = _op

    def func(self: "Quantity", other: Any) -> "Quantity":
        if isinstance(other, units.UnitBase):
            return Quantity(self.value, _unit_op(self.unit, other))
        return Quantity(
            _op(self.value, other.value), _unit_op(self.unit, other.unit)
        )

    return func


@register_pytree_node_class
class Quantity:
    __array_priority__ = 10001

    def __init__(self, value: Any, unit: Any):
        self.value = jnp.asarray(value)
        self.unit = units.Unit(unit)

    def __repr__(self) -> str:
        return f"Quantity(value={self.value}, unit={self.unit})"

    # # This interface doesn't currently play nice with PyTrees (see
    # # https://github.com/google/jax/issues/10065 for a related issue), but
    # # perhaps someday something like this could work. Let's keep an eye on it!
    # # Don't forget to add dtype, shape, and size properties.
    # def __jax_array__(self) -> Any:
    #     try:
    #         factor = self.unit.to(units.dimensionless_unscaled)
    #     except units.UnitConversionError:
    #         raise ValueError(
    #             "Cannot convert Quantity to array unless dimensionless"
    #         )
    #     return factor * self.value

    def tree_flatten(self) -> Tuple[Tuple[Any], Any]:
        children = (self.value,)
        aux_data = self.unit
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Tuple[Any]) -> "Quantity":
        return cls(*children, unit=aux_data)

    def to(self, unit: Any, equivalencies: Iterable[Any] = []) -> "Quantity":
        new_unit = units.Unit(unit)
        if new_unit is self.unit:
            return self
        factor = self.unit.to(new_unit, equivalencies=equivalencies)
        return Quantity(factor * self.value, new_unit)

    __add__ = _impl_additive_binary(operator.add)
    __radd__ = _impl_additive_binary(operator.add, reflect=True)
    __sub__ = _impl_additive_binary(operator.sub)
    __rsub__ = _impl_additive_binary(operator.sub, reflect=True)
    __mul__ = _impl_multiplicative_binary(operator.mul)
    __rmul__ = _impl_multiplicative_binary(operator.mul, reflect=True)
    __truediv__ = _impl_multiplicative_binary(operator.truediv)
    __rtruediv__ = _impl_multiplicative_binary(operator.truediv, reflect=True)
    __matmul__ = _impl_multiplicative_binary(
        operator.matmul, unit_op=operator.mul
    )
    __rmatmul__ = _impl_multiplicative_binary(
        operator.matmul, unit_op=operator.mul, reflect=True
    )
