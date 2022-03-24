"""Module containing tools to calculate statistics of collective events.

Example:
    >>> from arcos4py.tools import calcCollevStats
    >>> test = calcCollevStats()
    >>> out = test().run(data=data,frame_column = "frame", collid_column = "collid")
"""

import pandas as pd


class calcCollevStats:
    """Class to calculate statistics of collective events."""

    def __init__(self) -> None:
        """Class to calculate statistics of collective events."""
        pass

    def _calculate_duration_size_group(self, data: pd.DataFrame, frame_column: str) -> pd.DataFrame:
        """Calculates duration and size for the collective event in the dataframe.

        Parameters:
            data: pandas dataframe
                filtered dataframe containing a single collective event

            frame_column: str
                string indicating the contained frame column

        Returns:
            Dataframe containing duration, tot_size, min_size and
            max_size of the current collective event
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

        Parameters:
            data: pandas dataframe
                dataframe containing unfiltered collective events

            collev_id: str
                string indicating the contained collective id column

            frame_column: str
                string indicating the contained frame column

        Returns:
            Dataframe containing duration and tot_size of all collective events

        """
        data_gp = data.groupby([collev_id])
        colev_duration = data_gp.apply(lambda x: self._calculate_duration_size_group(x, frame_column))
        colev_duration = colev_duration.droplevel(-1).reset_index()
        return colev_duration

    def calculate(self, data: pd.DataFrame, frame_column: str, collid_column: str) -> pd.DataFrame:
        """Calculate statistics of collective events.

        Parameters:
            data: pandas dataframe
                filtered dataframe containing a single collective event

            frame_column: str
                string indicating the frame column in data

            collid_column: str
                string indicating the collective event id column in data

        Returns:
            pandas dataframe containing collective events stats
        """
        colev_stats = self._get_collev_duration(data, frame_column, collid_column)
        return colev_stats
