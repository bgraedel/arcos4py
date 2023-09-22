"""Module containing clipping and interpolation classes.

Example:
    >>> # Interpolation
    >>> from arcos4py.tools import interpolation
    >>> a = interpolation(data)
    >>> data_interp = a.interpolate()

    >>> # clipping
    >>> from arcos4py.tools import clipMeas
    >>> a = clipMeas(data)
    >>> data_clipped = a.clip(0.001, 0.999)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter
from skimage.util import view_as_blocks


class interpolation:
    """Interpolate nan values in a numpy array.

    Attributes:
        data (DataFrame): Where NaN should be replaced with interpolated values.
    """

    def __init__(self, data: pd.DataFrame):
        """Interpolate nan values in a pandas dataframe.

        Uses pandas.interpolate with liner interpolation.

        Arguments:
            data (DataFrame): Where NaN should be replaced with interpolated values.
        """
        self.data = data

    def interpolate(self) -> pd.DataFrame:
        """Interpolate nan and missing values.

        Returns:
            DataFrame: Interpolated input data.
        """
        self.data = self.data.interpolate(axis=0)

        return self.data


class clipMeas:
    """Clip input array."""

    def __init__(self, data: np.ndarray) -> None:
        """Clips array to quantilles.

        Arguments:
            data (ndarray): To be clipped.
        """
        self.data = data

    def _calculate_percentile(
        self, data: np.ndarray, clip_low: float, clip_high: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate upper and lower quantille.

        Arguments:
            data (ndarray): To calculate upper and lower quantile on.
            clip_low (float): Lower clipping boundary (quantile).
            clip_high (float): Upper clipping boundry (quantille).

        Returns:
            np.ndarray: Array with lower quantile and array with upper quantile.

        """
        quantille_low = np.quantile(data, clip_low, keepdims=True)
        quantille_high = np.quantile(data, clip_high, keepdims=True)
        return quantille_low, quantille_high

    def clip(self, clip_low: float = 0.001, clip_high: float = 0.999) -> np.ndarray:
        """Clip input array to upper and lower quantiles defined in clip_low and clip_high.

        Arguments:
            clip_low (float): Lower clipping boundary (quantile).
            clip_high (float): Upper clipping boundry (quantille).

        Returns:
            np.ndarray (np.ndarray): A clipped array of the input data.
        """
        low, high = self._calculate_percentile(self.data, clip_low, clip_high)
        out = self.data.clip(low, high)
        return out


def remove_image_background(
    image: np.ndarray, filter_type: str = 'gaussian', size=(10, 1, 1), dims="TXY", crop_time_axis: bool = False
) -> np.ndarray:
    """Removes background from images. Assumes axis order (t, y, x) for 2d images and (t, z, y, x) for 3d images.

    Arguments:
        image (np.ndarray): Image to remove background from.
        filter_type (Union[str, function]): Filter to use to remove background. Can be one of ['median', 'gaussian'].
        size (int, Tuple): Size of filter to use. For median filter, this is the size of the window.
            For gaussian filter, this is the standard deviation.
            If a single int is passed in, it is assumed to be the same for all dimensions.
            If a tuple is passed in, it is assumed to correspond to the size of the filter in each dimension.
            Default is (10, 1, 1).
        dims (str): Dimensions to apply filter over. Can be one of ['TXY', 'TZXY']. Default is 'TXY'.
        crop_time_axis (bool): Whether to crop the time axis. Default is True.
    Returns (np.ndarray): Image with background removed.
        Along the first axis (t) half of the filter size is removed from the beginning and end respectively.
    """
    # correct images with a filter applied over time
    allowed_filters = ["median", "gaussian"]
    dims_list = list(dims.upper())

    # check input
    for i in dims_list:
        if i not in dims_list:
            raise ValueError(f"Invalid dimension {i}. Must be 'T', 'X', 'Y', or 'Z'.")

    if len(dims_list) > len(set(dims_list)):
        raise ValueError("Duplicate dimensions in dims.")

    if len(dims_list) != image.ndim:
        raise ValueError(
            f"Length of dims must be equal to number of dimensions in image. Image has {image.ndim} dimensions."
        )
    # make sure axis dont occur twice and that they are valid
    if len(dims) != len(set(dims)):
        raise ValueError('Dimensions must not occur twice.')

    if filter_type not in allowed_filters:
        raise ValueError(f'Filter type must be one of {allowed_filters}.')

    # get index of time axis
    t_idx = dims_list.index("T")

    orig_image = image.copy()

    if isinstance(size, int):
        size = (size,) * image.ndim
    elif isinstance(size, tuple):
        if len(size) != image.ndim:
            raise ValueError(f'Filter size must have {image.ndim} dimensions.')
        # check size of dimensions are compatible with image
        for idx, s in enumerate(size):
            if s > image.shape[idx]:
                raise ValueError(f'Filter size in dimension {idx} is larger than image size in that dimension.')
    else:
        raise ValueError('Filter size must be an int or tuple.')

    if filter_type == 'median':
        filtered = median_filter(orig_image, size=size)
    elif filter_type == 'gaussian':
        filtered = gaussian_filter(orig_image, sigma=size)

    # crop time axis if necessary
    shift = size[t_idx] // 2
    corr = np.subtract(orig_image, filtered, dtype=np.float32)
    if crop_time_axis:
        corr = corr[shift:-shift]

    return corr


def blockwise_median(a, blockshape):
    """Calculates the blockwise median of an array.

    Arguments:
        a (np.ndarray): Array to calculate blockwise median of.
        blockshape (Tuple): Shape of blocks to use.
    Returns (np.ndarray): Blockwise median of array.
    """
    assert a.ndim == len(blockshape), "blocks must have same dimensionality as the input image"
    assert not (np.array(a.shape) % blockshape).any(), "blockshape must divide cleanly into the input image shape"

    block_view = view_as_blocks(a, blockshape)
    assert block_view.shape[a.ndim :] == blockshape
    block_axes = [*range(a.ndim, 2 * a.ndim)]
    return np.median(block_view, axis=block_axes)
