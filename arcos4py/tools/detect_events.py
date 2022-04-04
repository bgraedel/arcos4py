"""Module to track and detect collective events.

Example:
    >>> from arcos4py.tools import detectCollev
    >>> ts = detectCollev(data)
    >>> events_df = ts.run()
"""

from typing import Union

import pandas as pd
from numpy import int64
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from ._errors import columnError, epsError, minClSzError, noDataError, nPrevError


class detectCollev:
    """Identifies and tracks collective signalling events.

    Requires binarized measurement column.
    Makes use of the dbscan algorithm,
    applies this to every timeframe and subsequently connects
    collective events between frames located within eps distance of each other.

    Attributes:
        input_data (DataFrame): Input data to be processed. Must contain a binarized measurement column.
        eps (float): The maximum distance between two samples for one to be considered as in
            the neighbourhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            Value is also used to connect collective events across multiple frames.
        minClSz (int): Minimum size for a cluster to be identified as a collective event.
        nPrev (int): Number of previous frames the tracking
            algorithm looks back to connect collective events.
        posCols (list): List of position columns contained in the data.
            Must at least contain one
        frame_column (str): Indicating the frame column in input_data.
        id_column (str): Indicating the track id/id column in input_data.
        bin_meas_column (str): Indicating the bin_meas_column in input_data or None.
        clid_column (str): Indicating the column name containing the ids of collective events.
    """

    def __init__(
        self,
        input_data: pd.DataFrame,
        eps: float = 1,
        minClSz: int = 1,
        nPrev: int = 1,
        posCols: list = ["x"],
        frame_column: str = 'time',
        id_column: str = 'id',
        bin_meas_column: Union[str, None] = 'meas',
        clid_column: str = 'clTrackID',
    ) -> None:
        """Constructs class with input parameters.

        Arguments:
            input_data (DataFrame): Input data to be processed. Must contain a binarized measurement column.
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighbourhood of the other.
                This is not a maximum bound on the distances of points within a cluster.
                Value is also used to connect collective events across multiple frames.
            minClSz (int): Minimum size for a cluster to be identified as a collective event.
            nPrev (int): Number of previous frames the tracking
                algorithm looks back to connect collective events.
            posCols (list): List of position columns contained in the data.
                Must at least contain one
            frame_column (str): Indicating the frame column in input_data.
            id_column (str): Indicating the track id/id column in input_data.
            bin_meas_column (str): Indicating the bin_meas_column in input_data or None.
            clid_column (str): Indicating the column name containing the ids of collective events.
        """
        # assign some variables passed in as arguments to the object
        self.input_data = input_data
        self.eps = eps
        self.minClSz = minClSz
        self.nPrev = nPrev
        self.frame_column = frame_column
        self.id_column = id_column
        self.bin_meas_column = bin_meas_column
        self.clid_column = clid_column
        self.posCols = posCols
        self.columns_input = self.input_data.columns
        self.clidFrame = f'{clid_column}.frame'

        self.pos_cols_inputdata = [col for col in self.posCols if col in self.columns_input]

        # run input checks
        self._run_input_checks()

    def _check_input_data(self):
        """Checks if input contains data\
        raises error if not."""
        if self.input_data is None:
            raise noDataError("Input is None")
        elif self.input_data.empty:
            raise noDataError("Input is empty")

    def _check_pos_columns(self):
        """Checks if Input contains correct columns\
        raises Exception if not."""
        if not all(item in self.columns_input for item in self.posCols):
            raise columnError("Input data does not have the indicated position columns!")

    def _check_frame_column(self):
        if self.frame_column not in self.columns_input:
            raise columnError("Input data does not have the indicated frame column!")

    def _check_id_column(self):
        if self.id_column not in self.columns_input:
            raise columnError("Input data does not have the indicated id column!")

    def _check_eps(self):
        """Checks if eps is greater than 0."""
        if self.eps <= 0:
            raise epsError("eps has to be greater than 0")

    def _check_minClSz(self):
        if self.minClSz <= 0:
            raise minClSzError("Parameter eps has to be greater than 0!")

    def _check_nPrev(self):
        if self.nPrev <= 0:
            raise nPrevError("Parameter nPrev has to be an integer greater than 0!")

    def _run_input_checks(self):
        """Run input checks."""
        self._check_input_data()
        self._check_pos_columns()
        self._check_eps()
        self._check_minClSz()
        self._check_nPrev()
        self._check_frame_column()

    def _select_necessary_columns(
        self, data: pd.DataFrame, frame_col: str, id_col: str, pos_col: list, bin_col: Union[str, None]
    ) -> pd.DataFrame:
        """Select necessary input colums from input data into dataframe.

        Arguments:
            data (DataFrame): Containing necessary columns.
            frame_col (str): Frame column in data.
            id_col (str): Id column in data.
            pos_col (list): string representation of position columns in data.
            bin_col (str): Name of binary column.

        Returns (DataFrame):
            Filtered columns necessary for calculation.
        """
        if bin_col is None:
            columns = [frame_col, id_col]
        else:
            columns = [frame_col, id_col, bin_col]
        columns.extend(pos_col)
        neccessary_data = data[columns].copy(deep=True)
        return neccessary_data

    def _filter_active(self, data: pd.DataFrame, bin_meas_col: Union[str, None]) -> pd.DataFrame:
        """Selects rows with binary value of greater than 0.

        Arguments:
            data (DataFrame): Dataframe containing necessary columns.
            bin_meas_col (str|None): Either name of the binary column or None if no such column exists.

        Returns (DataFrame):
            Filtered pandas DataFrame.
        """
        if bin_meas_col is not None:
            data = data[data[bin_meas_col] > 0]
        return data

    def _dbscan(self, x: pd.DataFrame, collid_col: str) -> pd.DataFrame:
        """Dbscan method to run and merge the cluster id labels to the original dataframe.

        Arguments:
            x (DataFrame): With unique frame and position columns.
            collid_col (str): Column to be created containing cluster-id labels.

        Returns (Dataframe):
            Dataframe with added collective id column detected by DBSCAN.
        """
        pos_array = x[self.pos_cols_inputdata]
        db_array = DBSCAN(eps=self.eps, min_samples=self.minClSz, algorithm="kd_tree").fit(pos_array)
        cluster_labels = db_array.labels_
        cluster_list = [id + 1 for id in cluster_labels.tolist()]
        x[collid_col] = cluster_list
        x = x[x[collid_col] > 0]
        return x

    def _run_dbscan(self, data: pd.DataFrame, frame: str, clid_frame: str) -> pd.DataFrame:
        """Apply dbscan method to every group i.e. frame.

        Arguments:
            data (DataFrame): Must contain position columns and frame columns.
            frame (str): Name of frame column in data.
            clid_frame (str): column to be created containing the output cluster ids from dbscan.

        Returns (Dataframe):
            Dataframe with added collective id column detected by DBSCAN for every frame.
        """
        data_gb = data.groupby([frame])
        db_labels = data_gb.apply(lambda x: self._dbscan(x, clid_frame))
        db_labels = db_labels.reset_index(drop=True)
        return db_labels

    def _make_db_id_unique(self, db_data: pd.DataFrame, frame: str, clid_frame, clid) -> pd.DataFrame:
        """Make db_scan cluster id labels unique by adding the\
        cummulative sum of previous group to next group.

        Arguments:
            db_data (DataFrame): Returned by _run_dbscan function with non-unique cluster ids.
            frame (str): Frame column.
            clid_frame (str): Column name of cluster-id per frame.
            clid (str): Column name of unique cluster ids to be returned.

        Returns (Dataframe):
            Dataframe with unique collective events.
        """
        db_data_n = db_data[[clid_frame, frame]]
        db_gp = db_data_n.groupby([frame])
        db_max = db_gp.max().reset_index()
        db_max["PreviouMax"] = db_max[clid_frame].shift(1).fillna(0)
        db_max["PreviouMax_cumsum"] = db_max["PreviouMax"].cumsum()
        db_data = db_max[[frame, "PreviouMax_cumsum"]].merge(db_data, on=frame)
        db_data[self.clidFrame] += db_data["PreviouMax_cumsum"]
        db_data = db_data.drop(columns=["PreviouMax_cumsum"])
        db_data[clid] = db_data[clid_frame].astype(int64)
        return db_data

    def _nearest_neighbour(
        self,
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        nbr_nearest_neighbours: int = 1,
    ):
        """Calculates nearest neighbour in from data_a\
        to data_b nearest_neighbours in data_b.

        Arguments:
            data_a (DataFrame): containing position values.
            data_b (DataFrame): containing position values.
            nbr_nearest_neighbours (int): of the number of nearest neighbours to be calculated.

        Returns (tuple):
            Returns tuple of 2 arrays containing nearest neighbour indices and distances.
        """
        kdB = KDTree(data=data_a.values)
        nearest_neighbours = kdB.query(data_b.values, k=nbr_nearest_neighbours)
        return nearest_neighbours

    def _link_clusters_between_frames(self, data: pd.DataFrame, frame: str, colid: str) -> pd.DataFrame:
        """Tracks clusters detected with DBSCAN along a frame axis,\
        returns tracked collective events as a pandas dataframe.

        Arguments:
            data (DataFrame): Output from dbscan.
            frame (str): Frame column.
            colid (str): Colid column.

        Returns (Dataframe):
            Pandas dataframe with tracked collective ids.
        """
        # loop over all frames to link detected clusters iteratively
        for t in sorted(data[frame].unique())[1:]:
            prev_frame = data[(data[frame] >= (t - self.nPrev)) & (data[frame] < (t))].copy(deep=True)
            current_frame = data[data[frame] == t].copy(deep=True)
            # only continue if objects were detected in previous frame
            if not prev_frame.empty:
                colid_current = current_frame[colid]
                # loop over unique cluster in frame
                for cluster in sorted(colid_current.unique()):
                    pos_current = current_frame[self.posCols][current_frame[colid] == cluster]

                    pos_previous = prev_frame[self.posCols]
                    # calculate nearest neighbour between previoius and current frame
                    nn_dist, nn_indices = self._nearest_neighbour(pos_previous, pos_current)
                    prev_cluster_nbr_all = prev_frame.iloc[nn_indices][colid]
                    prev_cluster_nbr_eps = prev_cluster_nbr_all[nn_dist <= self.eps]
                    # only continue if neighbours
                    # were detected within eps distance
                    if not prev_cluster_nbr_eps.empty:
                        prev_clusternbr_eps_unique = prev_cluster_nbr_eps.unique()
                        colid_subset = data[(data[frame] == t) & (data[colid] == cluster)][colid]
                        subset_index_list = list(colid_subset.index.values)
                        # propagate cluster id from previous frame
                        # if multiple clusters in the eps of nearest neighbour
                        if len(prev_clusternbr_eps_unique) > 0:
                            data.loc[subset_index_list, colid] = prev_cluster_nbr_all.values

        data[colid] = data.groupby(colid).ngroup() + 1
        return data

    def _get_export_columns(self):
        """Get columns that will contained in the pandas dataframe returned by the run method."""
        self.pos_cols_inputdata = [col for col in self.posCols if col in self.columns_input]
        columns = [self.frame_column, self.id_column]
        columns.extend(self.pos_cols_inputdata)
        columns.append(self.clid_column)
        return columns

    def run(self) -> pd.DataFrame:
        """Method to execute the different steps necessary for tracking.

        1. Selects columns.
        2. filters data on binary column > 1.
        3. Applies dbscan algorithm to every frame.
        4. Makes cluster ids unique across frames.
        5. Tracks collective events i.e. links cluster ids across frames.
        6. Creates final DataFrame.

        Returns (Dataframe):
            Dataframe with tracked collective events is returned.
        """
        filtered_cols = self._select_necessary_columns(
            self.input_data,
            self.frame_column,
            self.id_column,
            self.pos_cols_inputdata,
            self.bin_meas_column,
        )
        active_data = self._filter_active(filtered_cols, self.bin_meas_column)
        db_data = self._run_dbscan(
            data=active_data,
            frame=self.frame_column,
            clid_frame=self.clidFrame,
        )
        db_data = self._make_db_id_unique(
            db_data,
            frame=self.frame_column,
            clid_frame=self.clidFrame,
            clid=self.clid_column,
        )
        tracked_events = self._link_clusters_between_frames(db_data, self.frame_column, self.clid_column)
        return_columns = self._get_export_columns()
        tracked_events = tracked_events[return_columns]
        if self.clid_column in self.input_data.columns:
            df_to_merge = self.input_data.drop(columns=[self.clid_column])
        else:
            df_to_merge = self.input_data
        tracked_events = tracked_events.merge(df_to_merge, how="left")
        return tracked_events
