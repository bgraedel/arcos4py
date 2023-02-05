"""Arcos4py top level module.

This package is a python implementation of the Arcos algorithm for the detection
and tracking of collective events intime-series data.
"""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.1.6'

from ._arcos4py import ARCOS

__all__ = ["ARCOS"]
