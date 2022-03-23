"""Module to track and detect collective events."""

from typing import Union

import pandas as pd

from .tools.binarize_detrend import binData
from .tools.cleandata import clipMeas, interpolation
from .tools.detect_events import detectCollev


class ARCOS:
    """Detects and tracks collective events in a tracked timeseries dataset.

    Requires binarized measurment column, that can be generated with the
    bin_measurements method.
    Tracking uses of the dbscan algorithm, applys this to every timeframe
    and subsequently connects collective events between frames located
    within eps distance of each other.

    Class methods:
    -------------

    interpolate_measurements(),
    clip_meas(),
    bin_measurements(),
    trackCollev()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        posCols: list = ["x"],
        frame_column: str = 'time',
        id_column: str = 'id',
        measurment_column: str = 'meas',
        clid_column: str = 'clTrackID',
    ) -> None:
        """Detects and tracks collective events in a tracked timeseries dataset.

        Parameters
        ----
        data: pandas DataFrame
            Input Data of tracked timeseries containing position columns, a measurment column and an object ID column

        posCols: list
            dict containing positin column names (strings) inside data e.g. At least one dimension is required
            >>> posCols = ['posx', 'posy']

        frame_column: str
            String indicating the frame column in input_data

        id_column: str
            String indicating the track id/id column in input_data

        measurment_column: str
            String indicating the measurment column in input_data

        clid_column: str
            String indicating the column name containing the collective event ids
        """
        self.data = data
        self.posCols = posCols
        self.frame_column = frame_column
        self.id_column = id_column
        self.measurment_column = measurment_column
        self.clid_column = clid_column

        self.data_binarized: pd.DataFrame = None
        self.tracked_events: pd.DataFrame = None
        self.bin_col: Union[str, None] = None
        # to check if no measurment was provided assign None

        self.data = self.data.sort_values(by=[self.frame_column, self.id_column])
        self._check_col_dict()
        if self.measurment_column is not None:
            self.resc_col = f"{self.measurment_column}.resc"
            self.bin_col = f"{self.measurment_column}.bin"

    def __repr__(self):
        """Returns self.data when calling print() on self."""
        return repr(self.data)

    def _check_col_dict(self):
        """Checks that self.cols contains all required columns."""
        columns = self.data.columns
        input_columns = [self.frame_column, self.id_column, self.id_column, self.measurment_column]
        input_columns = [col for col in input_columns if col is not None]
        if not all(item in columns for item in input_columns):
            raise ValueError(f"Columns {input_columns} do not match with column in dataframe.")

    def interpolate_measurements(self) -> pd.DataFrame:
        """Interpolates measurment column NaN's in place.

        Returns
        -------
            Dataframe with interpolated measurment column
        """
        meas_column = self.data[self.measurment_column].to_numpy()
        meas_interp = interpolation(meas_column).interpolate()
        self.data[self.measurment_column] = meas_interp
        return self.data

    def clip_meas(self, clip_low: float = 0.001, clip_high=0.999) -> pd.DataFrame:
        """Clip measurment column to upper and lower quantilles defined in clip_low and clip_high.

        Args
        ----
        clip_low: float
        lower clipping boundry (quantille)

        clip_high: float
        upper clipping boundry (quantille)

        Returns
        -------
        Dataframe with inplace clipped measurment column
        """
        meas_column = self.data[self.measurment_column].to_numpy()
        meas_clipped = clipMeas(meas_column).clip(clip_low, clip_high)
        self.data[self.measurment_column] = meas_clipped
        return self.data

    def bin_measurements(
        self,
        smoothK: int = 3,
        biasK: int = 51,
        peakThr: float = 0.2,
        binThr: float = 0.1,
        polyDeg: int = 1,
        biasMet: str = "runmed",
    ) -> pd.DataFrame:
        r"""Smooth, de-trend, and binarise the input data.

        First a short-term median filter with size smoothK
        is applied to remove fast noise from the time series.
        If the de-trending method is set to \code{"none"},
        smoothing is applied on globally rescaled time series.
        The subsequent de-trending can be performed with a long-term median filter
        with the size biasK {biasMet = "runmed"}
        or by fitting a polynomial of degree polyDeg {biasMet = "lm"}.

        After de-trending,
        if the global difference between min/max is greater than the threshold
        the signal is rescaled to the (0,1) range.
        The final signal is binarised using the binThr threshold

        Args
        ----
        smoothK: int, default = 3
            Size of the short-term median smoothing filter.

        biasK: int, default = 51
            Size of the long-term de-trending median filter

        peakThr: float, default = 0.2
            Threshold for rescaling of the de-trended signal.

        polyDeg: int, default = 1
            Sets the degree of the polynomial for lm fitting.

        biasMet: str
            De-trending method, one of ['runmed', 'lm', 'none'].

        Returns
        -------
        Dataframe with detrended/smoothed and binarized measurment column
        """
        self.data = binData(
            self.data,
            smoothK,
            biasK,
            peakThr,
            binThr,
            polyDeg,
            biasMet,
            colMeas=self.measurment_column,
            colGroup=self.id_column,
        ).run()
        return self.data

    def trackCollev(self, eps: float = 1, minClsz: int = 1, nPrev: int = 1) -> pd.DataFrame:
        """Identifies and tracks collective signalling events.

        Requires binarized measurment column.
        Makes use of the dbscan algorithm,
        applys this to every timeframe and subsequently connects
        collective events between frames located within eps distance of each other.

        Args
        ----
        eps: float
            The maximum distance between two samples for one to be considered as in
            the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            Value also used to connect collective events across multiple frames.

        minClSz: int
            Minimum size for a cluster to be identified as a collective event

        nPrev: int
            Number of previous frames the tracking
            algorithm looks back to connect collective events

        Returns
        -------
        pandas dataframe with detected collective events across time

        """
        self.data = detectCollev(
            self.data,
            eps=eps,
            minClSz=minClsz,
            nPrev=nPrev,
            posCols=self.posCols,
            frame_column=self.frame_column,
            id_column=self.id_column,
            bin_meas_column=self.bin_col,
            clid_column=self.clid_column,
        ).run()

        return self.data
