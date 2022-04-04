"""Module containing tools to calculate statistics of collective events.

Example:
    >>> from arcos4py.tools import calcCollevStats
    >>> test = calcCollevStats()
    >>> out = test().run(data = data,frame_column = "frame", collid_column = "collid")
"""

import pandas as pd


class calcCollevStats:
    """Class to calculate statistics of collective events."""

    def __init__(self) -> None:
        """Class to calculate statistics of collective events."""
        pass

    def _calculate_duration_size_group(self, data: pd.DataFrame, frame_column: str) -> pd.DataFrame:
        """Calculates duration and size for the collective event in the dataframe.

        Arguments:
            data (DataFrame): Containing a single collective event.
            frame_column (str): Indicating the contained frame column.

        Returns (Dataframe):
            Dataframe containing duration, tot_size, min_size and
            max_size of the current collective event.
        """
        coll_dur = max(data[frame_column]) - min(data[frame_column]) + 1
        coll_total_size = data[frame_column].size
        coll_min_size = min(data.groupby([frame_column]).size())
        coll_max_size = max(data.groupby([frame_column]).size())
        d = {
            "duration": [coll_dur],
            "tot_size": [coll_total_size],
            "min_size": [coll_min_size],
            "max_size": [coll_max_size],
        }
        combined_df = pd.DataFrame(d)
        return combined_df

    def _get_collev_duration(
        self,
        data: pd.DataFrame,
        frame_column: str,
        collev_id: str,
    ) -> pd.DataFrame:
        """Applies self._calculate_duration_size_group() to every group\
        i.e. every collective event.

        Arguments:
            data (DataFrame): Containing unfiltered collective events.
            collev_id (str): Indicating the contained collective id column.
            frame_column (str): Indicating the contained frame column.

        Returns (DataFrame):
            DataFrame containing duration and tot_size of all collective events.

        """
        data_gp = data.groupby([collev_id])
        colev_duration = data_gp.apply(lambda x: self._calculate_duration_size_group(x, frame_column))
        colev_duration = colev_duration.droplevel(-1).reset_index()
        return colev_duration

    def calculate(self, data: pd.DataFrame, frame_column: str, collid_column: str) -> pd.DataFrame:
        """Calculate statistics of collective events.

        Arguments:
            data (DataFrame): Containing collective events.
            frame_column (str): Indicating the frame column in data.
            collid_column (str): Indicating the collective event id column in data.

        Returns:
            Dataframe containing collective events stats.
        """
        if data.empty:
            return data
        colev_stats = self._get_collev_duration(data, frame_column, collid_column)
        return colev_stats
