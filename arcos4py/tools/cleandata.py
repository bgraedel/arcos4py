"""Module containing clipping and interplation classes."""
import numpy as np


class interpolation:
    """Interpolate nan values in a numpy array.

    Parameters
    ----
    data: np.ndarray
        nd array, where nan should be replaced with interpolated values
    """

    def __init__(self, data: np.ndarray):
        """Interpolate nan values in a numpy array.

        Parameters
        ----
        data: np.ndarray
            nd array, where nan should be replaced with interpolated values
        """
        self.data = data

    def _nan_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """
        return np.isnan(y), lambda z: z.nonzero()[0]

    def interpolate(self):
        """Interpolate nan and missing values.

        Args
        ---
        data: np.ndarray
            input data, should be 1 or 2d np

        Returns
        -------
        interpolated input data
        """
        nans, x = self._nan_helper(self.data)
        self.data[nans] = np.interp(x(nans), x(~nans), self.data[~nans])
        return self.data


class clipMeas:
    """Clips array to quantilles.

    Parameters
    ----
    data: numpy ndarray
    """

    def __init__(self, data: np.ndarray):
        """Clips array to quantilles.

        Parameters
        -----
        data: numpy ndarray
            input array to be clipped
        """
        self.data = data

    def _calculate_percentile(self, data: np.ndarray, clip_low: float, clip_high: float):
        """Calculate upper and lower quantille.

        Args
        ----
        clip_low: float
        lower clipping boundry (quantille)

        clip_high: float
        upper clipping boundry (quantille)

        Returns
        -------
        array with lower quantille, array with upper quantille

        """
        quantille_low = np.quantile(data, clip_low, keepdims=True)
        quantille_high = np.quantile(data, clip_high, keepdims=True)
        return quantille_low, quantille_high

    def clip(self, clip_low: float = 0.001, clip_high: float = 0.999):
        """Clip input array to upper and lower quantilles defined in clip_low and clip_high.

        Args
        ----
        clip_low: float
        lower clipping boundry (quantille)

        clip_high: float
        upper clipping boundry (quantille)

        Returns
        -------
        clipped array of input data
        """
        low, high = self._calculate_percentile(self.data, clip_low, clip_high)
        out = self.data.clip(low, high)
        return out


if __name__ == "__main__":
    y = np.array([1, 1, 1, np.NaN, np.NaN, 2, 2, np.NaN, 0])
    int_dta = interpolation(y).interpolate()
    test = clipMeas(int_dta).clip(clip_low=0.1, clip_high=0.8)
    print(test)
