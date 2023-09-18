"""Tools for detecting collective events."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = '0.2.1'

from ._binarize_detrend import binData
from ._cleandata import clipMeas, interpolation
from ._detect_events import detectCollev, estimate_eps, track_events_dataframe, track_events_image
from ._filter_events import filterCollev
from ._stats import calcCollevStats, calculate_statistics, calculate_per_frame_statistics

__all__ = [
    "binData",
    "clipMeas",
    "interpolation",
    "detectCollev",
    "filterCollev",
    "calcCollevStats",
    "estimate_eps",
    "track_events_dataframe",
    "track_events_image",
    "calculate_statistics",
]
