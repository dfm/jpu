# -*- coding: utf-8 -*-
"""
JAX + Units
===========
"""

__all__ = [
    "numpy",
    "monkey",
    "is_quantity",
    "grad",
    "value_and_grad",
    "UnitRegistry",
]

from jpu import monkey, numpy
from jpu.core import grad, is_quantity, value_and_grad
from jpu.numpy_helper import implement_numpy_functions
from jpu.registry import UnitRegistry

implement_numpy_functions()
del implement_numpy_functions

__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__uri__ = "https://github.com/dfm/jpu"
__license__ = "MIT"
__description__ = "JAX + Units"
__copyright__ = "2022 Simons Foundation, Inc"
__contributors__ = "https://github.com/dfm/jpu/graphs/contributors"
__bibtex__ = __citation__ = """TBD"""
