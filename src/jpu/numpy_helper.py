# mypy: ignore-errors

import sys
from functools import wraps
from itertools import chain

import jax.numpy as jnp
from pint import numpy_func


# Based closely on the pint implementation:
# https://github.com/hgrecco/pint/blob/37a61ede6fbd628c7dc160eb36278cf41c96484c/pint/facets/numpy/numpy_func.py#L244
def _implement_func(func_name, input_units=None, output_unit=None):
    jax_func = getattr(jnp, func_name)

    @wraps(jax_func)
    def wrapped(*args, **kwargs):
        try:
            first_input_units = numpy_func._get_first_input_units(args, kwargs)
        except TypeError:
            # None of the inputs are quantities
            return jax_func(*args, **kwargs)
        if input_units == "all_consistent":
            # Match all input args/kwargs to same units
            (
                stripped_args,
                stripped_kwargs,
            ) = numpy_func.convert_to_consistent_units(
                *args, pre_calc_units=first_input_units, **kwargs
            )
        else:
            if isinstance(input_units, str):
                # Conversion requires Unit, not str
                pre_calc_units = first_input_units._REGISTRY.parse_units(
                    input_units
                )
            else:
                pre_calc_units = input_units

            # Match all input args/kwargs to input_units, or if input_units is None,
            # simply strip units
            (
                stripped_args,
                stripped_kwargs,
            ) = numpy_func.convert_to_consistent_units(
                *args, pre_calc_units=pre_calc_units, **kwargs
            )

        # Determine result through plain numpy function on stripped arguments
        result_magnitude = jax_func(*stripped_args, **stripped_kwargs)

        if output_unit is None:
            # Short circuit and return magnitude alone
            return result_magnitude
        elif output_unit == "match_input":
            result_unit = first_input_units
        elif output_unit in [
            "sum",
            "mul",
            "delta",
            "delta,div",
            "div",
            "invdiv",
            "variance",
            "square",
            "sqrt",
            "cbrt",
            "reciprocal",
            "size",
        ]:
            result_unit = numpy_func.get_op_output_unit(
                output_unit,
                first_input_units,
                tuple(chain(args, kwargs.values())),
            )
        else:
            result_unit = output_unit

        return first_input_units._REGISTRY.Quantity(
            result_magnitude, result_unit
        )

    module = sys.modules["jpu.numpy"]
    module.__all__.append(func_name)
    setattr(module, func_name, wrapped)


def implement_numpy_functions():
    for func_name in numpy_func.strip_unit_input_output_ufuncs:
        _implement_func(func_name)

    for func_name in numpy_func.matching_input_bare_output_ufuncs:
        _implement_func(func_name, input_units="all_consistent")

    for (
        func_name,
        out_unit,
    ) in numpy_func.matching_input_set_units_output_ufuncs.items():
        _implement_func(
            func_name, input_units="all_consistent", output_unit=out_unit
        )

    for func_name, (in_unit, out_unit) in numpy_func.set_units_ufuncs.items():
        _implement_func(func_name, input_units=in_unit, output_unit=out_unit)

    for func_name in numpy_func.matching_input_copy_units_output_ufuncs:
        _implement_func(
            func_name, input_units="all_consistent", output_unit="match_input"
        )

    for func_name in numpy_func.copy_units_output_ufuncs:
        _implement_func(func_name, output_unit="match_input")

    for func_name, unit_op in numpy_func.op_units_output_ufuncs.items():
        _implement_func(func_name, output_unit=unit_op)

    for func_name in ["cumprod", "cumproduct", "nancumprod"]:
        _implement_func(func_name, input_units="", output_unit="")

    for func_name in ["block", "hstack", "vstack", "dstack", "column_stack"]:
        _implement_func(
            func_name, input_units="all_consistent", output_unit="match_input"
        )

    for func_name in [
        "size",
        "isreal",
        "iscomplex",
        "shape",
        "ones_like",
        "zeros_like",
        "empty_like",
        "argsort",
        "argmin",
        "argmax",
        "alen",
        "ndim",
        "nanargmax",
        "nanargmin",
        "count_nonzero",
        "nonzero",
        "result_type",
    ]:
        _implement_func(func_name)

    for unit_type, funcs in {
        "sum": ["std", "nanstd", "sum", "nansum", "cumsum", "nancumsum"],
        "mul": ["cross", "trapz", "dot"],
        "delta": ["diff", "ediff1d"],
        "delta,div": ["gradient"],
        "variance": ["var", "nanvar"],
    }.items():
        for func_name in funcs:
            _implement_func(func_name, output_unit=unit_type)
