"""Tools for validating detected collective events."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.1.5'

from .bootstrapping import bootstrap_arcos
from .resampling import resample_data

__all__ = [
    "resample_data",
    "bootstrap_arcos",
]
