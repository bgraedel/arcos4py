"""Module to plot different metrics generated by arcos4py functions.

Example:
    >>> from arcos4py.plotting import arcosPlots
    >>> plt = arcosPlots(data, 'time', 'meas')
    >>> plt.plot_detrended()
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class arcosPlots:
    """Plot different arcos metrics such as detrended vs original data."""

    def __init__(self, data: pd.DataFrame, frame: str, measurement: str, detrended: str, id: str):
        """Plot different arcos metrics such as detrended vs original data.

        Arguments:
            data: Dataframe
                containing ARCOS data.

            frame: String
                name of frame column in data.

            measurement: String
                name of measurement column in data.

            detrended: String
                name of detrended column with detrended data.

            id: String
                name of track id column.
        """
        self.data = data
        self.measurement = measurement
        self.detrended = detrended
        self.id = id
        self.frame = frame

    def plot_detrended(
        self, n_samples: int = 25, subplots: tuple = (5, 5), plotsize: tuple = (20, 10)
    ) -> matplotlib.axes.Axes:
        """Method to plot detrended vs original data.

        Arguments:
            n_samples: int,
                Number of tracks to plot.

            subplots:
                Number of subplots, should be approx. one per sample.

            plotsize:
                Size of generated plot.

        Returns:
            Matplotlib figure of detrended vs original data.

        """
        vals = np.random.choice(self.data[self.id].unique(), n_samples, replace=False)
        self.data = self.data.set_index(self.id).loc[vals].reset_index()
        grouped = self.data.groupby(self.id)

        ncols = subplots[0]
        nrows = subplots[1]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=plotsize, sharey=True)

        for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
            grouped.get_group(key).plot(x=self.frame, y=[self.measurement, self.detrended], ax=ax)
            ax.get_legend().remove()

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right")
        plt.show()