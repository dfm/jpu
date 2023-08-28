"""When imported, this submodule implements all of the functions in
``jpu.numpy`` closely following the logic implemented in
``pint.facets.numpy.numpy_func``. These implemented functions are then injected
into the ``jpu.numpy`` namespace in the appropriate places.

Since JAX doesn't support any sort of array dispatch protocol, you'll need to
use the functions defined in ``jpu.numpy`` instead of ``jax.numpy`` to get
support for units.
"""

from functools import wraps
from inspect import signature
from itertools import chain

import jax.numpy as jnp
from pint.facets.numpy import numpy_func

from jpu import numpy as jpu_numpy
from jpu.numpy import linalg as linalg

HANDLED_FUNCTIONS = {}


def implements(numpy_func_string):
    def decorator(func):
        # Unlike in the pint implementation, we assign all functions to the jpu.numpy
        # module
        func_str_split = numpy_func_string.split(".")
        func_name = func_str_split[-1]

        # Get the appropriate jpu.numpy submodule
        module = jpu_numpy
        for func_str_piece in func_str_split[:-1]:
            module = getattr(module, func_str_piece)

        # Extract the jax.numpy function that we can fall back to
        jax_func = getattr(jnp, func_str_split[0])
        for func_str_piece in func_str_split[1:]:
            jax_func = getattr(jax_func, func_str_piece)

        # Unlike the pint implementation, we fall back to the jnp implementation
        # when none of the inputs are Quantities
        @wraps(func)
        def wrapped(*args, **kwargs):
            # TODO(dfm): This could maybe just check args, because we typically
            # assume that the args will have units... Are there any cases where a
            # quantity in kwargs would be valid?
            if not any(map(numpy_func._is_quantity, chain(args, kwargs.values()))):
                return jax_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Save this wrapped function to the jpu.numpy module
        if hasattr(module, func_name):
            print(f"Function {func_name} has already been implemented")
        setattr(module, func_name, wrapped)

        # The rest is the same as the pint implementation
        HANDLED_FUNCTIONS[numpy_func_string] = wrapped

        return wrapped

    return decorator


def implement_func(func_str, input_units=None, output_unit=None):
    func_str_split = func_str.split(".")
    func = getattr(jnp, func_str_split[0], None)
    for func_str_piece in func_str_split[1:]:
        func = getattr(func, func_str_piece)

    @implements(func_str)
    def implementation(*args, **kwargs):
        first_input_units = numpy_func._get_first_input_units(args, kwargs)

        if input_units == "all_consistent":
            stripped_args, stripped_kwargs = numpy_func.convert_to_consistent_units(
                *args, pre_calc_units=first_input_units, **kwargs
            )
        else:
            if isinstance(input_units, str):
                pre_calc_units = first_input_units._REGISTRY.parse_units(input_units)
            else:
                pre_calc_units = input_units

            stripped_args, stripped_kwargs = numpy_func.convert_to_consistent_units(
                *args, pre_calc_units=pre_calc_units, **kwargs
            )

        result_magnitude = func(*stripped_args, **stripped_kwargs)

        if output_unit is None:
            return result_magnitude
        elif output_unit == "match_input":
            result_unit = first_input_units
        elif output_unit in (
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
        ):
            result_unit = numpy_func.get_op_output_unit(
                output_unit, first_input_units, tuple(chain(args, kwargs.values()))
            )
        else:
            result_unit = output_unit

        return first_input_units._REGISTRY.Quantity(result_magnitude, result_unit)


# Unlike the pint implementation, we don't explicitly distinguish between ufuncs
# and functions, therefore some of the functions are commented out here because
# they are also implemented below as functions
function_specs = [
    # ** ****** **
    # ** UFUNCS **
    # ** ****** **
    #
    # strip input and output
    ("isnan", None, None),
    ("isinf", None, None),
    ("isfinite", None, None),
    ("signbit", None, None),
    ("sign", None, None),
    # bare output
    ("equal", "all_consistent", None),
    ("greater", "all_consistent", None),
    ("greater_equal", "all_consistent", None),
    ("less", "all_consistent", None),
    ("less_equal", "all_consistent", None),
    ("not_equal", "all_consistent", None),
    # matching input, set output
    ("arctan2", "all_consistent", "radian"),
    # set input and output
    # ("cumprod", "", ""),
    ("arccos", "", "radian"),
    ("arcsin", "", "radian"),
    ("arctan", "", "radian"),
    ("arccosh", "", "radian"),
    ("arcsinh", "", "radian"),
    ("arctanh", "", "radian"),
    ("exp", "", ""),
    ("expm1", "", ""),
    ("exp2", "", ""),
    ("log", "", ""),
    ("log10", "", ""),
    ("log1p", "", ""),
    ("log2", "", ""),
    ("sin", "radian", ""),
    ("cos", "radian", ""),
    ("tan", "radian", ""),
    ("sinh", "radian", ""),
    ("cosh", "radian", ""),
    ("tanh", "radian", ""),
    ("radians", "degree", "radian"),
    ("degrees", "radian", "degree"),
    ("deg2rad", "degree", "radian"),
    ("rad2deg", "radian", "degree"),
    ("logaddexp", "", ""),
    ("logaddexp2", "", ""),
    # matching input, copy output
    # ("compress", "all_consistent", "match_input"),
    ("conj", "all_consistent", "match_input"),
    ("conjugate", "all_consistent", "match_input"),
    # ("copy", "all_consistent", "match_input"),
    # ("diagonal", "all_consistent", "match_input"),
    # ("max", "all_consistent", "match_input"),
    # ("mean", "all_consistent", "match_input"),
    # ("min", "all_consistent", "match_input"),
    # ("ptp", "all_consistent", "match_input"),
    # ("ravel", "all_consistent", "match_input"),
    # ("repeat", "all_consistent", "match_input"),
    # ("reshape", "all_consistent", "match_input"),
    # ("round", "all_consistent", "match_input"),
    # ("squeeze", "all_consistent", "match_input"),
    # ("swapaxes", "all_consistent", "match_input"),
    # ("take", "all_consistent", "match_input"),
    ("trace", "all_consistent", "match_input"),
    # ("transpose", "all_consistent", "match_input"),
    ("ceil", "all_consistent", "match_input"),
    ("floor", "all_consistent", "match_input"),
    ("hypot", "all_consistent", "match_input"),
    ("rint", "all_consistent", "match_input"),
    ("copysign", "all_consistent", "match_input"),
    ("nextafter", "all_consistent", "match_input"),
    ("trunc", "all_consistent", "match_input"),
    ("absolute", "all_consistent", "match_input"),
    ("positive", "all_consistent", "match_input"),
    ("negative", "all_consistent", "match_input"),
    ("maximum", "all_consistent", "match_input"),
    ("minimum", "all_consistent", "match_input"),
    ("fabs", "all_consistent", "match_input"),
    # copy input to output
    ("ldexp", None, "match_input"),
    ("fmod", None, "match_input"),
    ("mod", None, "match_input"),
    ("remainder", None, "match_input"),
    # output operation on input
    ("var", None, "square"),
    ("multiply", None, "mul"),
    ("true_divide", None, "div"),
    ("divide", None, "div"),
    ("floor_divide", None, "div"),
    ("sqrt", None, "sqrt"),
    ("cbrt", None, "cbrt"),
    ("square", None, "square"),
    ("reciprocal", None, "reciprocal"),
    ("std", None, "sum"),
    ("sum", None, "sum"),
    ("cumsum", None, "sum"),
    ("matmul", None, "mul"),
    #
    # ** ********* **
    # ** FUNCTIONS **
    # ** ********* **
    #
    # matching input, copy output
    ("block", "all_consistent", "match_input"),
    ("hstack", "all_consistent", "match_input"),
    ("vstack", "all_consistent", "match_input"),
    ("dstack", "all_consistent", "match_input"),
    ("column_stack", "all_consistent", "match_input"),
    ("broadcast_arrays", "all_consistent", "match_input"),
    # strip input and output
    ("size", None, None),
    ("isreal", None, None),
    ("iscomplex", None, None),
    ("shape", None, None),
    ("ones_like", None, None),
    ("zeros_like", None, None),
    ("empty_like", None, None),
    ("argsort", None, None),
    ("argmin", None, None),
    ("argmax", None, None),
    ("ndim", None, None),
    ("nanargmax", None, None),
    ("nanargmin", None, None),
    ("count_nonzero", None, None),
    ("nonzero", None, None),
    ("result_type", None, None),
    # output operation on input
    # ("std", None, "sum"),
    ("nanstd", None, "sum"),
    # ("sum", None, "sum"),
    ("nansum", None, "sum"),
    # ("cumsum", None, "sum"),
    ("nancumsum", None, "sum"),
    ("diff", None, "delta"),
    ("ediff1d", None, "delta"),
    ("gradient", None, "delta,div"),
    ("linalg.solve", None, "invdiv"),
    # ("var", None, "variance"),
    ("nanvar", None, "variance"),
]

for func_str, input_units, output_unit in function_specs:
    implement_func(func_str, input_units=input_units, output_unit=output_unit)


@implements("modf")
def _modf(x, *args, **kwargs):
    (x,), output_wrap = numpy_func.unwrap_and_wrap_consistent_units(x)
    return tuple(output_wrap(y) for y in jnp.modf(x, *args, **kwargs))


@implements("frexp")
def _frexp(x, *args, **kwargs):
    (x,), output_wrap = numpy_func.unwrap_and_wrap_consistent_units(x)
    mantissa, exponent = jnp.frexp(x, *args, **kwargs)
    return output_wrap(mantissa), exponent


@implements("power")
def _power(x1, x2):
    if numpy_func._is_quantity(x1):
        return x1**x2

    return x2.__rpow__(x1)


@implements("add")
def _add(x1, x2, *args, **kwargs):
    (x1, x2), output_wrap = numpy_func.unwrap_and_wrap_consistent_units(x1, x2)
    return output_wrap(jnp.add(x1, x2, *args, **kwargs))


@implements("subtract")
def _subtract(x1, x2, *args, **kwargs):
    (x1, x2), output_wrap = numpy_func.unwrap_and_wrap_consistent_units(x1, x2)
    return output_wrap(jnp.subtract(x1, x2, *args, **kwargs))


@implements("meshgrid")
def _meshgrid(*xi, **kwargs):
    input_units = (x.units for x in xi)
    res = jnp.meshgrid(*(x.m for x in xi), **kwargs)
    return [out * unit for out, unit in zip(res, input_units)]


@implements("full_like")
def _full_like(a, fill_value, **kwargs):
    if hasattr(fill_value, "_REGISTRY"):
        return fill_value._REGISTRY.Quantity(
            jnp.ones_like(a, **kwargs) * fill_value.m,
            fill_value.units,
        )

    return jnp.ones_like(a, **kwargs) * fill_value


@implements("interp")
def _interp(x, xp, fp, left=None, right=None, period=None):
    (x, xp, period), _ = numpy_func.unwrap_and_wrap_consistent_units(x, xp, period)
    (fp, right, left), output_wrap = numpy_func.unwrap_and_wrap_consistent_units(
        fp, left, right
    )
    return output_wrap(jnp.interp(x, xp, fp, left=left, right=right, period=period))


@implements("concatenate")
def _concatenate(sequence, *args, **kwargs):
    sequence, output_wrap = numpy_func.unwrap_and_wrap_consistent_units(*sequence)
    return output_wrap(jnp.concatenate(sequence, *args, **kwargs))


@implements("stack")
def _stack(arrays, *args, **kwargs):
    arrays, output_wrap = numpy_func.unwrap_and_wrap_consistent_units(*arrays)
    return output_wrap(jnp.stack(arrays, *args, **kwargs))


@implements("unwrap")
def _unwrap(p, discont=None, axis=-1):
    # np.unwrap only dispatches over p argument, so assume it is a Quantity
    discont = jnp.pi if discont is None else discont
    return p._REGISTRY.Quantity(
        jnp.unwrap(p.m_as("rad"), discont, axis=axis), "rad"
    ).to(p.units)


@implements("einsum")
def _einsum(subscripts, *operands, **kwargs):
    operand_magnitudes, _ = numpy_func.convert_to_consistent_units(
        *operands, pre_calc_units=None
    )
    output_unit = numpy_func.get_op_output_unit(
        "mul", numpy_func._get_first_input_units(operands), operands
    )
    return jnp.einsum(subscripts, *operand_magnitudes, **kwargs) * output_unit


@implements("isin")
def _isin(element, test_elements, assume_unique=False, invert=False):
    if not numpy_func._is_quantity(element):
        raise ValueError(
            "Cannot test if unit-aware elements are in not-unit-aware array"
        )

    if numpy_func._is_quantity(test_elements):
        try:
            test_elements = test_elements.m_as(element.units)
        except numpy_func.DimensionalityError:
            # Incompatible unit test elements cannot be in element
            return jnp.full(element.shape, False)
    elif not element.dimensionless:
        # Unit do not match, so all false
        return jnp.full(element.shape, False)
    else:
        # Convert to units of element
        element._REGISTRY.Quantity(test_elements).m_as(element.units)

    return jnp.isin(
        element.m, test_elements, assume_unique=assume_unique, invert=invert
    )


@implements("pad")
def _pad(array, pad_width, mode="constant", **kwargs):
    def _recursive_convert(arg, unit):
        if not numpy_func._is_quantity(arg):
            arg = unit._REGISTRY.Quantity(arg, "dimensionless")
        return arg.m_as(unit)

    # pad only dispatches on array argument, so we know it is a Quantity
    units = array.units

    # Handle flexible constant_values and end_values, converting to units if Quantity
    # and ignoring if not
    for key in ("constant_values", "end_values"):
        if key in kwargs:
            kwargs[key] = _recursive_convert(kwargs[key], units)

    return units._REGISTRY.Quantity(
        jnp.pad(array._magnitude, pad_width, mode=mode, **kwargs), units
    )


def _require_multiplicative(func):
    @wraps(func)
    def wrapped(a, *args, **kwargs):
        if numpy_func._is_quantity(a) and not a._is_multiplicative:
            raise ValueError("Boolean value of Quantity with offset unit is ambiguous.")

        return func(a, *args, **kwargs)

    return wrapped


@implements("where")
@_require_multiplicative
def _where(condition, *args):
    condition = getattr(condition, "magnitude", condition)
    args, output_wrap = numpy_func.unwrap_and_wrap_consistent_units(*args)
    return output_wrap(jnp.where(condition, *args))


@implements("any")
@_require_multiplicative
def _any(a, *args, **kwargs):
    return jnp.any(a._magnitude, *args, **kwargs)


@implements("all")
@_require_multiplicative
def _all(a, *args, **kwargs):
    return jnp.all(a._magnitude, *args, **kwargs)


def implement_prod_func(name):
    func = getattr(jnp, name, None)

    @implements(name)
    def _prod(a, *args, **kwargs):
        arg_names = ("axis", "dtype", "out", "keepdims", "initial", "where")
        all_kwargs = dict(**dict(zip(arg_names, args)), **kwargs)
        axis = all_kwargs.get("axis", None)
        where = all_kwargs.get("where", None)

        registry = a.units._REGISTRY

        if axis is not None and where is not None:
            raise NotImplementedError
        elif axis is not None:
            units = a.units ** a.shape[axis]
        elif where is not None:
            exponent = jnp.sum(where)
            units = a.units**exponent
        else:
            exponent = (
                jnp.sum(jnp.logical_not(jnp.isnan(a))) if name == "nanprod" else a.size
            )
            units = a.units**exponent

        result = func(a._magnitude, *args, **kwargs)

        return registry.Quantity(result, units)


for name in ("prod", "nanprod"):
    implement_prod_func(name)


@implements("trapz")
def _trapz(a, x=None, dx=1.0, **kwargs):
    a = numpy_func._base_unit_if_needed(a)
    units = a.units
    if x is not None:
        if hasattr(x, "units"):
            x = numpy_func._base_unit_if_needed(x)
            units *= x.units
            x = x._magnitude
        ret = jnp.trapz(a._magnitude, x, **kwargs)
    else:
        if hasattr(dx, "units"):
            dx = numpy_func._base_unit_if_needed(dx)
            units *= dx.units
            dx = dx._magnitude
        ret = jnp.trapz(a._magnitude, dx=dx, **kwargs)

    return a.units._REGISTRY.Quantity(ret, units)


def implement_mul_func(func):
    func = getattr(jnp, func_str)

    @implements(func_str)
    def implementation(a, b, **kwargs):
        a = numpy_func._base_unit_if_needed(a)
        units = a.units
        if hasattr(b, "units"):
            b = numpy_func._base_unit_if_needed(b)
            units *= b.units
            b = b._magnitude

        mag = func(a._magnitude, b, **kwargs)
        return a.units._REGISTRY.Quantity(mag, units)


for func_str in ("cross", "dot"):
    implement_mul_func(func_str)


def implement_consistent_units_by_argument(func_str, unit_arguments, wrap_output=True):
    if "." not in func_str:
        func = getattr(jnp, func_str, None)
    else:
        parts = func_str.split(".")
        module = jnp
        for part in parts[:-1]:
            module = getattr(module, part, None)
        func = getattr(module, parts[-1], None)

    @implements(func_str)
    def implementation(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        valid_unit_arguments = [
            label
            for label in unit_arguments
            if label in bound_args.arguments and bound_args.arguments[label] is not None
        ]
        unwrapped_unit_args, output_wrap = numpy_func.unwrap_and_wrap_consistent_units(
            *(bound_args.arguments[label] for label in valid_unit_arguments)
        )
        for i, unwrapped_unit_arg in enumerate(unwrapped_unit_args):
            bound_args.arguments[valid_unit_arguments[i]] = unwrapped_unit_arg
        ret = func(*bound_args.args, **bound_args.kwargs)

        if wrap_output:
            return output_wrap(ret)
        return ret


for func_str, unit_arguments, wrap_output in (
    ("expand_dims", "a", True),
    ("squeeze", "a", True),
    ("rollaxis", "a", True),
    ("moveaxis", "a", True),
    ("around", "a", True),
    ("diagonal", "a", True),
    ("mean", "a", True),
    ("ptp", "a", True),
    ("ravel", "a", True),
    ("repeat", "a", True),
    ("round_", "a", True),
    ("round", "a", True),
    ("take", "a", True),
    ("sort", "a", True),
    ("median", "a", True),
    ("nanmedian", "a", True),
    ("transpose", "a", True),
    ("copy", "a", True),
    ("average", "a", True),
    ("nanmean", "a", True),
    ("swapaxes", "a", True),
    ("nanmin", "a", True),
    ("nanmax", "a", True),
    ("percentile", "a", True),
    ("nanpercentile", "a", True),
    ("quantile", "a", True),
    ("nanquantile", "a", True),
    ("flip", "m", True),
    ("fix", "x", True),
    ("trim_zeros", ["filt"], True),
    ("broadcast_to", ["array"], True),
    ("amax", ["a", "initial"], True),
    ("amin", ["a", "initial"], True),
    ("max", ["a", "initial"], True),
    ("min", ["a", "initial"], True),
    ("searchsorted", ["a", "v"], False),
    ("nan_to_num", ["x", "nan", "posinf", "neginf"], True),
    ("clip", ["a", "a_min", "a_max"], True),
    ("append", ["arr", "values"], True),
    ("compress", "a", True),
    ("linspace", ["start", "stop"], True),
    ("tile", "A", True),
    # ("lib.stride_tricks.sliding_window_view", "x", True),
    ("rot90", "m", True),
    ("insert", ["arr", "values"], True),
    ("delete", ["arr"], True),
    ("resize", "a", True),
    ("reshape", "a", True),
    ("intersect1d", ["ar1", "ar2"], True),
):
    implement_consistent_units_by_argument(func_str, unit_arguments, wrap_output)


def implement_close(func_str):
    func = getattr(jnp, func_str)

    @implements(func_str)
    def implementation(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        labels = ["a", "b"]
        arrays = {label: bound_args.arguments[label] for label in labels}
        if "atol" in bound_args.arguments:
            atol = bound_args.arguments["atol"]
            a = arrays["a"]
            if not hasattr(atol, "_REGISTRY") and hasattr(a, "_REGISTRY"):
                atol_ = a._REGISTRY.Quantity(atol, a.units)
            else:
                atol_ = atol
            arrays["atol"] = atol_

        args, _ = numpy_func.unwrap_and_wrap_consistent_units(*arrays.values())
        for label, value in zip(arrays.keys(), args):
            bound_args.arguments[label] = value

        return func(*bound_args.args, **bound_args.kwargs)


for func_str in ("isclose", "allclose"):
    implement_close(func_str)


def implement_atleast_nd(func_str):
    func = getattr(jnp, func_str)

    @implements(func_str)
    def implementation(*arrays):
        stripped_arrays, _ = numpy_func.convert_to_consistent_units(*arrays)
        arrays_magnitude = func(*stripped_arrays)
        if len(arrays) > 1:
            return [
                array_magnitude
                if not hasattr(original, "_REGISTRY")
                else original._REGISTRY.Quantity(array_magnitude, original.units)
                for array_magnitude, original in zip(arrays_magnitude, arrays)
            ]
        else:
            output_unit = arrays[0].units
            return output_unit._REGISTRY.Quantity(arrays_magnitude, output_unit)


for func_str in ("atleast_1d", "atleast_2d", "atleast_3d"):
    implement_atleast_nd(func_str)


def implement_single_dimensionless_argument_func(func_str):
    func = getattr(jnp, func_str)

    @implements(func_str)
    def implementation(a, *args, **kwargs):
        (a_stripped,), _ = numpy_func.convert_to_consistent_units(
            a, pre_calc_units=a._REGISTRY.parse_units("dimensionless")
        )
        return a._REGISTRY.Quantity(func(a_stripped, *args, **kwargs))


for func_str in (
    "cumprod",
    # "cumproduct",  # deprecated
    "nancumprod",
):
    implement_single_dimensionless_argument_func(func_str)


# ** ***************************** **
# ** FUNCTIONS NOT COVERED BY PINT **
# ** ***************************** **


@implements("argpartition")
def _argpartition(a, *args, **kwargs):
    (a,), output_wrap = numpy_func.unwrap_and_wrap_consistent_units(a)
    return output_wrap(jnp.argpartition(a, *args, **kwargs))


@implements("choose")
def _choose(a, *args, **kwargs):
    (a,), output_wrap = numpy_func.unwrap_and_wrap_consistent_units(a)
    return output_wrap(jnp.choose(a, *args, **kwargs))
