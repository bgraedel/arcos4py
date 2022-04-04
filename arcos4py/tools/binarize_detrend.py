"""Module containing binarization and detrending classes.

Example:
    >>> from arcos4py.tools import binData
    >>> binarizer = binData(x=data, biasMet="lm", colMeas="ERK_KTR", colGroup="trackID", polyDeg=1)
    >>> data_rescaled = binarizer.run()
"""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, minmax_scale


class detrender:
    """Smooth and de-trend input data.

    First, a short-term median filter with size smoothK is applied
    to remove fast noise from the time series.
    The subsequent de-trending can be performed with a long-term median filter
    with the size biasK {biasMet = "runmed"}
    or by fitting a polynomial of degree polyDeg {biasMet = "lm"}.

    Attributes:
        x (DataFrame): Time series data for smoothing.
        smoothK (int): Representing the size of the short-term median smoothing filter.
        biasK (int): Representing the size of the long-term de-trending median filter.
        peakThr (float): Threshold for rescaling of the de-trended signal.
        polyDeg (int): Sets the degree of the polynomial for lm fitting.
        biasMet (str): Indicating de-trending method, one of ['runmed', 'lm', 'none'].
        colMeas (str): Name of measurement column in x.
        colGroup (str): Name of id column in x.
    """

    def __init__(
        self,
        x: pd.DataFrame,
        smoothK: int = 3,
        biasK: int = 51,
        peakThr: float = 0.2,
        polyDeg: int = 1,
        biasMet: str = "runmed",
        colMeas: str = "meas",
        colGroup: str = "id",
    ) -> None:
        """Smooth and de-trend input data.

        Arguments:
            x (DataFrame): Time series data for smoothing.
            smoothK (int): Representing the size of the short-term median smoothing filter.
            biasK (int): Representing the size of the long-term de-trending median filter.
            peakThr (float): Threshold for rescaling of the de-trended signal.
            polyDeg (int): Sets the degree of the polynomial for lm fitting.
            biasMet (str): Indicating de-trending method, one of ['runmed', 'lm', 'none'].
            colMeas (str): Name of measurement column in x.
            colGroup (str): Name of id column in x.
        """
        # check if biasmethod contains one of these three types
        biasMet_types = ["runmed", "lm", "none"]
        if biasMet not in biasMet_types:
            raise ValueError(f"Invalid bias method. Expected one of: {biasMet_types}")

        self.x = x
        self.smoothK = smoothK
        self.biasK = biasK
        self.peakThr = peakThr
        self.polyDeg = polyDeg
        self.biasMet = biasMet
        self.colMeas = colMeas
        self.colGroup = colGroup

    def _detrend_runnmed(self, data, filter_size, endrule_mode):
        local_smoothing = median_filter(input=data, size=filter_size, mode=endrule_mode)
        return local_smoothing

    def _detrend_lm(self, data, polynomial_degree):
        x = np.linspace(1, data.size, data.size).astype(int).reshape((-1, 1))
        transformer = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        data_ = transformer.fit_transform(x)
        model = LinearRegression().fit(X=data_, y=data)
        predicted_value = model.predict(data_)
        return predicted_value

    def _run_detrend(self, data: pd.DataFrame, col_meas: str) -> pd.DataFrame:
        if not data.empty:
            local_smoothed = self._detrend_runnmed(data[col_meas], self.smoothK, "constant")
            if self.biasMet != "none":
                if self.biasMet == "runmed":
                    global_smoothed = self._detrend_runnmed(
                        data=local_smoothed,
                        filter_size=self.biasK,
                        endrule_mode="constant",
                    )
                elif self.biasMet == "lm":
                    global_smoothed = self._detrend_lm(local_smoothed, self.polyDeg)

                local_smoothed = np.subtract(local_smoothed, global_smoothed)
                local_smoothed = np.clip(local_smoothed, 0, None)
                if (local_smoothed.max() - local_smoothed.min()) > self.peakThr:
                    local_smoothed = np.divide(local_smoothed, local_smoothed.max())
                local_smoothed = np.nan_to_num(local_smoothed)
        else:
            local_smoothed = None

        if not col_meas.endswith(".resc"):
            col_meas = f"{col_meas}.resc"
        data[col_meas] = local_smoothed
        return data

    def detrend(self, data: pd.DataFrame, group_col: str, resc_col) -> pd.DataFrame:
        """Run detrinding on input data.

        The method applies detrending to each group defined in group_col and
        outputs it into the resc_column.

        Returns (DataFrame): Dataframe containing rescaled column.
        """
        data_gp = data.groupby([group_col])
        data = data_gp.apply(lambda y: self._run_detrend(y, resc_col))
        return data


class binData(detrender):
    """Smooth, de-trend, and binarise the input data.

    First a short-term median filter with size smoothK
    is applied to remove fast noise from the time series.
    If the de-trending method is set to "none",
    smoothing is applied on globally rescaled time series.
    The subsequent de-trending can be performed with a long-term median filter
    with the size biasK {biasMet = "runmed"}
    or by fitting a polynomial of degree polyDeg {biasMet = "lm"}.

    After de-trending,
    if the global difference between min/max is greater than the threshold
    the signal is rescaled to the (0,1) range.
    The final signal is binarised using the binThr threshold.
    """

    def __init__(
        self,
        x: pd.DataFrame,
        smoothK: int = 3,
        biasK: int = 51,
        peakThr: float = 0.2,
        binThr: float = 0.1,
        polyDeg: int = 1,
        biasMet: str = "runmed",
        colMeas: str = "meas",
        colGroup: str = "id",
    ) -> None:
        """Smooth, de-trend, and binarise the input data.

        Arguments:
            x (DataFrame): The time-series data for smoothing.
            smoothK (int): Size of the short-term median smoothing filter.
            biasK (int): Size of the long-term de-trending median filter.
            peakThr (float): Threshold for rescaling of the de-trended signal.
            binThr (float): Threshold for binarizing the de-trended signal.
            polyDeg (int): Sets the degree of the polynomial for lm fitting.
            biasMet (str): De-trending method, one of ['runmed', 'lm', 'none'].
            colMeas (str): Measurement column in x.
            colGroup (str): Track id column in x.
        """
        super().__init__(x, smoothK, biasK, peakThr, polyDeg, biasMet, colMeas, colGroup)
        self.binThr = binThr
        self.col_resc = f"{self.colMeas}.resc"
        self.col_bin = f"{self.colMeas}.bin"

    def _rescale_data(self, df: pd.DataFrame, range: tuple = (0, 1)) -> pd.DataFrame:
        rescaled = minmax_scale(df[self.colMeas], feature_range=range)
        df[self.col_resc] = rescaled
        return df

    def _bin_data(self, df: pd.DataFrame) -> pd.DataFrame:
        bin = (df[self.col_resc].to_numpy() > self.binThr).astype(np.int_)
        df[self.col_bin] = bin
        return df

    def run(self) -> pd.DataFrame:
        """Runs binarization and detrending.

        If the bias Method is 'none', first it rescales the data to between [0,1], then
        local smoothing is applied to the measurement by groups, followed by
        binarization.

        If biasMeth is one of ['lm', 'runmed'], first the data is detrended locally with a
        median filter and then detrended globally, for 'lm' with a linear model and for 'runmed' with a
        median filter.
        Followed by binarization of the data.

        Returns (DataFrame): Dataframe containing Binarized data, rescaled data and the original columns
        """
        if self.biasMet == "none":
            rescaled_data = self._rescale_data(self.x)
            detrended_data = self.detrend(rescaled_data, self.colGroup, self.col_resc)
            binarized_data = self._bin_data(detrended_data)
        else:
            detrended_data = self.detrend(self.x, self.colGroup, self.colMeas)
            binarized_data = self._bin_data(detrended_data)

        return binarized_data
