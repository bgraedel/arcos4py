import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class arcosPlots:
    def __init__(self, data: pd.DataFrame, frame: str, measurment: str, detrended: str, id: str):
        self.data = data
        self.measurment = measurment
        self.detrended = detrended
        self.id = id
        self.frame = frame

    def plot_detrended(self, n_samples: int = 25, subplots: tuple = (5, 5), plotsize: tuple = (20, 10)):

        vals = np.random.choice(self.data[self.id].unique(), n_samples, replace=False)
        self.data = self.data.set_index(self.id).loc[vals].reset_index()
        grouped = self.data.groupby(self.id)

        ncols = subplots[0]
        nrows = subplots[1]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=plotsize, sharey=True)

        for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
            grouped.get_group(key).plot(x=self.frame, y=[self.measurment, self.detrended], ax=ax)
            ax.get_legend().remove()

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right")
        plt.show()
