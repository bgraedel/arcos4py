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
import numpy as np


class interpolation:
    """Interpolate nan values in a numpy array."""

    def __init__(self, data: np.ndarray):
        """Interpolate nan values in a numpy array.

        Arguments:
            data: np.ndarray,
                Where NaN should be replaced with interpolated values.
        """
        self.data = data

    def _nan_helper(self, y: np.ndarray):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs.

        Output:
            - nans, logical indices of NaNs.
            - index, a function, with signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices.

        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

        Returns:
            Returns interpolated input data.
        """
        return np.isnan(y), lambda z: z.nonzero()[0]

    def interpolate(self) -> np.ndarray:
        """Interpolate nan and missing values.

        Returns:
            Interpolated input data.
        """
        nans, x = self._nan_helper(self.data)
        self.data[nans] = np.interp(x(nans), x(~nans), self.data[~nans])
        return self.data


class clipMeas:
    """Clip input array."""

    def __init__(self, data: np.ndarray) -> None:
        """Clips array to quantilles.

        Arguments:
            data: np.ndarray,
                To be clipped.
        """
        self.data = data

    def _calculate_percentile(self, data: np.ndarray, clip_low: float, clip_high: float):
        """Calculate upper and lower quantille.

        Arguments:
            data: np.ndarray,
                To calculate upper and lower quantile on.

            clip_low: float,
                Lower clipping boundary (quantile).

            clip_high: float,
                Upper clipping boundry (quantille).

        Returns:
            Array with lower quantile and array with upper quantile.

        """
        quantille_low = np.quantile(data, clip_low, keepdims=True)
        quantille_high = np.quantile(data, clip_high, keepdims=True)
        return quantille_low, quantille_high

    def clip(self, clip_low: float = 0.001, clip_high: float = 0.999) -> np.ndarray:
        """Clip input array to upper and lower quantiles defined in clip_low and clip_high.

        Arguments:
            clip_low: float,
                Lower clipping boundary (quantile).

            clip_high: float,
                Upper clipping boundry (quantille).

        Returns:
            A clipped array of the input data.
        """
        low, high = self._calculate_percentile(self.data, clip_low, clip_high)
        out = self.data.clip(low, high)
        return out
