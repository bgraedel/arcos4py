"""Module to track and detect collective events.

Example:
    >>> from arcos4py.tools import detectCollev
    >>> ts = detectCollev(data)
    >>> events_df = ts.run()
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from kneed import KneeLocator
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from tqdm import auto


def _dbscan(x: np.ndarray, eps: float, minClSz: int) -> np.ndarray:
    """Dbscan method to run and merge the cluster id labels to the original dataframe.

    Arguments:
        x (np.ndarray): With unique frame and position columns.

    Returns:
        list[np.ndarray]: list with added collective id column detected by DBSCAN.
    """
    if x.size:
        db_array = DBSCAN(eps=eps, min_samples=minClSz, algorithm="kd_tree").fit(x)
        cluster_labels = db_array.labels_
        cluster_list = np.where(cluster_labels > -1, cluster_labels + 1, np.nan)
        return cluster_list

    return np.empty((0, 0))


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
        epsPrev (float | None): Frame to frame distance, value is used to connect
            collective events across multiple frames.If "None", same value as eps is used.
        minClSz (int): Minimum size for a cluster to be identified as a collective event.
        nPrev (int): Number of previous frames the tracking
            algorithm looks back to connect collective events.
        posCols (list): List of position columns contained in the data.
            Must at least contain one
        frame_column (str): Indicating the frame column in input_data.
        id_column (str | None): Indicating the track id/id column in input_data.
        bin_meas_column (str): Indicating the bin_meas_column in input_data or None.
        clid_column (str): Indicating the column name containing the ids of collective events.
    """

    def __init__(
        self,
        input_data: pd.DataFrame,
        eps: float = 1,
        epsPrev: Union[float, None] = None,
        minClSz: int = 1,
        nPrev: int = 1,
        posCols: list = ["x"],
        frame_column: str = 'time',
        id_column: Union[str, None] = None,
        bin_meas_column: Union[str, None] = 'meas',
        clid_column: str = 'clTrackID',
        n_jobs: int = 1,
    ) -> None:
        """Constructs class with input parameters.

        Arguments:
            input_data (DataFrame): Input data to be processed. Must contain a binarized measurement column.
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighbourhood of the other.
                This is not a maximum bound on the distances of points within a cluster.
            epsPrev (float | None): Frame to frame distance, value is used to connect
                collective events across multiple frames.If "None", same value as eps is used.
            minClSz (int): Minimum size for a cluster to be identified as a collective event.
            nPrev (int): Number of previous frames the tracking
                algorithm looks back to connect collective events.
            posCols (list): List of position columns contained in the data.
                Must at least contain one.
            frame_column (str): Indicating the frame column in input_data.
            id_column (str | None): Indicating the track id/id column in input_data, optional.
            bin_meas_column (str): Indicating the bin_meas_column in input_data or None.
            clid_column (str): Indicating the column name containing the ids of collective events.
            n_jobs (int): Number of paralell workers to spawn, -1 uses all available cpus.
        """
        # assign some variables passed in as arguments to the object
        self.input_data = input_data
        self.eps = eps
        if epsPrev:
            self.epsPrev = epsPrev
        else:
            self.epsPrev = eps
        self.minClSz = minClSz
        self.nPrev = nPrev
        self.frame_column = frame_column
        self.id_column = id_column
        self.bin_meas_column = bin_meas_column
        self.n_jobs = n_jobs
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
            raise ValueError("Input is None")
        elif self.input_data.empty:
            raise ValueError("Input is empty")

    def _check_pos_columns(self):
        """Checks if Input contains correct columns\
        raises Exception if not."""
        if not all(item in self.columns_input for item in self.posCols):
            raise ValueError("Input data does not have the indicated position columns!")

    def _check_frame_column(self):
        if self.frame_column not in self.columns_input:
            raise ValueError("Input data does not have the indicated frame column!")

    def _check_eps(self):
        """Checks if eps is greater than 0."""
        if self.eps <= 0:
            raise ValueError("Parameter eps has to be greater than 0")

    def _check_epsPrev(self):
        """Checks if frame to frame distance is greater than 0."""
        if self.epsPrev and self.epsPrev <= 0:
            raise ValueError("Parameter epsPrev has to be greater than 0 or None")

    def _check_minClSz(self):
        """Checks if minClSiz is greater than 0."""
        if self.minClSz <= 0:
            raise ValueError("Parameter minClSiz has to be an integer greater than 0!")

    def _check_nPrev(self):
        """Checks if nPrev is greater than 0."""
        if self.nPrev <= 0 and isinstance(self.nPrev, int):
            raise ValueError("Parameter nPrev has to be an integer greater than 0!")

    def _run_input_checks(self):
        """Run input checks."""
        self._check_input_data()
        self._check_pos_columns()
        self._check_eps()
        self._check_minClSz()
        self._check_nPrev()
        self._check_frame_column()

    def _select_necessary_columns(
        self,
        data: pd.DataFrame,
        frame_col: str,
        pos_col: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select necessary input colums from input data and returns them as numpy arrays.

        Arguments:
            data (DataFrame): Containing necessary columns.
            frame_col (str): Name of the frame/timepoint column in the dataframe.
            pos_col (list): string representation of position columns in data.

        Returns:
            np.ndarray, np.ndarray: Filtered columns necessary for calculation.
        """
        frame_column_np = data[frame_col].to_numpy()
        pos_columns_np = data[pos_col].to_numpy()
        return frame_column_np, pos_columns_np

    def _filter_active(self, data: pd.DataFrame, bin_meas_col: Union[str, None]) -> pd.DataFrame:
        """Selects rows with binary value of greater than 0.

        Arguments:
            data (DataFrame): Dataframe containing necessary columns.
            bin_meas_col (str|None): Either name of the binary column or None if no such column exists.

        Returns:
            DataFrame: Filtered pandas DataFrame.
        """
        if bin_meas_col is not None:
            data = data[data[bin_meas_col] > 0]
        return data

    # @profile
    def _run_dbscan(self, frame_data: np.ndarray, pos_data: np.ndarray, n_jobs: int = 1) -> list[np.ndarray]:
        """Apply dbscan method to every group i.e. frame. Assumes input data is sorted according to frame and id column.

        Arguments:
            frame_data (np.ndarray): frame column as a numpy array.
            pos_data (str): positions/coordinate columns as a numpy array.
            n_jobs (str): Number of workers to spawn, -1 uses all available cpus.

        Returns:
            list[np.ndarray]: list of arrays containing collective id column detected by DBSCAN for every frame.
        """
        assert frame_data.shape[0] == pos_data.shape[0]
        grouped_array = np.split(pos_data, np.unique(frame_data, axis=0, return_index=True)[1][1:])
        # map dbscan to grouped_array
        # out = [_dbscan(i) for i in grouped_array]
        out = Parallel(n_jobs=n_jobs)(
            delayed(_dbscan)(
                x=i,
                eps=self.eps,
                minClSz=self.minClSz,
            )
            for i in auto.tqdm(grouped_array)
        )
        return out

    def _make_db_id_unique(self, x: list[np.ndarray]) -> np.ndarray:
        """Make db_scan cluster id labels unique by adding the\
        cummulative sum of previous group to next group.

        Arguments:
            x (DataFrame): list of arrays containing collective event ids.

        Returns:
            np.ndarray: Unique collective event ids.
        """
        max_array = [0] + [np.nanmax(i) if i.size != 0 and not np.isnan(i).all() else 0 for i in x]
        max_array_cumsum = np.cumsum(np.nan_to_num(max_array))
        x_unique = [np.add(value, max_array_cumsum[i]) for i, value in enumerate(x)]
        out = np.concatenate(x_unique)
        return out

    # @profile
    def _link_clusters_between_frames(
        self, frame_data: np.ndarray, pos_data: np.ndarray, colid_data: np.ndarray, propagation_threshold: int = 1
    ) -> np.ndarray:
        """Tracks clusters detected with DBSCAN along a frame axis,\
        returns tracked collective events as a pandas dataframe.

        Arguments:
            frame_data (np.ndarray): Array containing frame/timepoint data.
            pos_data (np.ndarray): 2D array containing position coordinates of objects.
            colid_data (np.ndarray): Collective evetn ids
            propagation_threshold (int): Threshold for at least how many neighbours have to
                be within eps to propagate the cluster id.

        Returns:
            np.ndarray: Tracked collective event ids.
        """
        assert frame_data.shape[0] == pos_data.shape[0] == colid_data.shape[0]
        unique_frame_vals = np.unique(frame_data, return_index=False)
        # loop over all frames to link detected clusters iteratively
        for t in auto.tqdm(unique_frame_vals[1:]):
            prev_frame_mask = (frame_data >= (t - self.nPrev)) & (frame_data < t)
            current_frame_mask = frame_data == t
            prev_frame_colid = colid_data[prev_frame_mask]
            current_frame_colid = colid_data[current_frame_mask]
            prev_frame_pos_data = pos_data[prev_frame_mask]
            current_frame_pos_data = pos_data[current_frame_mask]
            kdtree_prevframe = KDTree(data=prev_frame_pos_data)

            # only continue if objects were detected in previous frame
            if prev_frame_colid.size:
                current_frame_colid_unique = np.unique(current_frame_colid, return_index=False)
                # loop over unique cluster in frame
                for cluster in current_frame_colid_unique:
                    pos_current_cluster = current_frame_pos_data[current_frame_colid == cluster]
                    # calculate nearest neighbour between previoius and current frame
                    nn_dist, nn_indices = kdtree_prevframe.query(pos_current_cluster, k=1)
                    prev_cluster_nbr_all = prev_frame_colid[nn_indices]
                    prev_cluster_nbr_eps = prev_cluster_nbr_all[(nn_dist <= self.epsPrev)]
                    # only continue if neighbours
                    # were detected within eps distance
                    if prev_cluster_nbr_eps.size >= propagation_threshold:
                        prev_clusternbr_eps_unique = np.unique(prev_cluster_nbr_eps, return_index=False)
                        if prev_clusternbr_eps_unique.size > 0:
                            # propagate cluster id from previous frame
                            colid_data[((current_frame_mask) & (colid_data == cluster))] = prev_cluster_nbr_all

        consecutive_collids = np.unique(colid_data, return_inverse=True)[1] + 1
        return consecutive_collids

    def _sort_input_dataframe(self, x: pd.DataFrame, frame_col: str, object_id_col: str | None) -> pd.DataFrame:
        """Sorts the input dataframe according to the frame column and track id column if available."""
        if object_id_col:
            x = x.sort_values([frame_col, object_id_col]).reset_index(drop=True)
        else:
            x = x.sort_values([frame_col]).reset_index(drop=True)
        return x

    def run(self, copy: bool = True) -> pd.DataFrame:
        """Method to execute the different steps necessary for tracking.

        Arguments:
            copy (bool): If True, the input data is copied before processing.
                If False, the input data is modified in place. Default is True.

        Returns:
            DataFrame: Dataframe with tracked collective events is returned.

        1. Selects columns.
        2. filters data on binary column > 1.
        3. Applies dbscan algorithm to every frame.
        4. Makes cluster ids unique across frames.
        5. Tracks collective events i.e. links cluster ids across frames.
        6. Creates final DataFrame.
        """
        if copy:
            x = self.input_data.copy()
        else:
            x = self.input_data
        x_sorted = self._sort_input_dataframe(x, frame_col=self.frame_column, object_id_col=self.id_column)
        x_filtered = self._filter_active(x_sorted, self.bin_meas_column)
        frame_data, pos_data = self._select_necessary_columns(
            x_filtered,
            self.frame_column,
            self.pos_cols_inputdata,
        )
        clid_vals = self._run_dbscan(frame_data=frame_data, pos_data=pos_data, n_jobs=self.n_jobs)
        clid_vals_unique = self._make_db_id_unique(
            clid_vals,
        )
        nan_rows = np.isnan(clid_vals_unique)
        tracked_events = self._link_clusters_between_frames(
            frame_data[~nan_rows], pos_data[~nan_rows], clid_vals_unique[~nan_rows]
        )

        if self.clid_column in x_sorted.columns:
            df_out = x_filtered.iloc[~nan_rows].drop(columns=[self.clid_column]).copy().reset_index(drop=True)
        else:
            df_out = x_filtered.iloc[~nan_rows].copy().reset_index(drop=True)
        # tracked_events = tracked_events.merge(df_to_merge, how="left")
        df_out[self.clid_column] = tracked_events
        return df_out


def _nearest_neighbour_eps(
    X: np.ndarray,
    nbr_nearest_neighbours: int = 1,
):
    kdB = KDTree(data=X)
    nearest_neighbours, indices = kdB.query(X, k=nbr_nearest_neighbours)
    return nearest_neighbours[:, 1:]


def estimate_eps(
    data: pd.DataFrame,
    method: str = 'kneepoint',
    pos_cols: list[str] = ['x,y'],
    frame_col: str = 't',
    n_neighbors: int = 5,
    plot: bool = True,
    plt_size: tuple[int, int] = (5, 5),
    **kwargs: dict,
):
    """Estimates eps parameter in DBSCAN.

    Estimates the eps parameter for the DBSCAN clustering method, as used by ARCOS,
    by calculating the nearest neighbour distances for each point in the data.
    N_neighbours should be chosen to match the minimum point size in DBSCAN
    or the minimum clustersize in detect_events respectively.
    The method argument determines how the eps parameter is estimated.
    'kneepoint' estimates the knee of the nearest neighbour distribution.
    'mean' and 'median' return (by default) 1.5 times
    the mean or median of the nearest neighbour distances respectively.

    Arguments:
        data (pd.DataFrame): DataFrame containing the data.
        method (str, optional): Method to use for estimating eps. Defaults to 'kneepoint'.
            Can be one of ['kneepoint', 'mean', 'median'].'kneepoint' estimates the knee of the nearest neighbour
            distribution to to estimate eps. 'mean' and 'median' use the 1.5 times the mean or median of the
            nearest neighbour distances respectively.
        pos_cols (list[str]): List of column names containing the position data.
        frame_col (str, optional): Name of the column containing the frame number. Defaults to 't'.
        n_neighbors (int, optional): Number of nearest neighbours to consider. Defaults to 5.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        plt_size (tuple[int, int], optional): Size of the plot. Defaults to (5, 5).
        kwargs: Keyword arguments for the method. Modify behaviour of respecitve method.
            For kneepoint: [S online, curve, direction, interp_method,polynomial_degree; For mean: [mean_multiplier]
            For median [median_multiplier]

    Returns:
        Eps (float): eps parameter for DBSCAN.
    """
    subset = [frame_col] + pos_cols
    for i in subset:
        if i not in data.columns:
            raise ValueError(f"Column {i} not in data")
    method_option = ['kneepoint', 'mean', 'median']

    if method not in method_option:
        raise ValueError(f"Method must be one of {method_option}")

    allowedtypes: dict[str, str] = {
        'kneepoint': 'kneepoint_values',
        'mean': 'mean_values',
        'median': 'median_values',
    }

    kwdefaults: dict[str, Any] = {
        'S': 1,
        'online': True,
        'curve': 'convex',
        'direction': 'increasing',
        'interp_method': 'polynomial',
        'mean_multiplier': 1.5,
        'median_multiplier': 1.5,
        'polynomial_degree': 7,
    }

    kwtypes: dict[str, Any] = {
        'S': int,
        'online': bool,
        'curve': str,
        'direction': str,
        'interp_method': str,
        'polynomial_degree': int,
        'mean_multiplier': (float, int),
        'median_multiplier': (float, int),
    }

    allowedkwargs: dict[str, list[str]] = {
        'kneepoint_values': ['S', 'online', 'curve', 'interp_method', 'direction', 'polynomial_degree'],
        'mean_values': ['mean_multiplier'],
        'median_values': ['median_multiplier'],
    }

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[method]]:
            raise ValueError(f'{key} keyword not in allowed keywords {allowedkwargs[allowedtypes[method]]}')
        if not isinstance(kwargs[key], kwtypes[key]):
            raise ValueError(f'{key} must be of type {kwtypes[key]}')

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[method]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    subset = [frame_col] + pos_cols
    data_np = data[subset].to_numpy(dtype=np.float64)
    # sort by frame
    data_np = data_np[data_np[:, 0].argsort()]
    grouped_array = np.split(data_np[:, 1:], np.unique(data_np[:, 0], axis=0, return_index=True)[1][1:])
    # map nearest_neighbours to grouped_array
    distances = np.concatenate([_nearest_neighbour_eps(i, n_neighbors) for i in grouped_array if i.shape[0] > 1])
    # flatten array
    distances_flat = distances.flatten()
    distances_flat = distances_flat[np.isfinite(distances_flat)]
    distances_sorted = np.sort(distances_flat)
    if method == 'kneepoint':
        k1 = KneeLocator(
            np.arange(0, distances_sorted.shape[0]),
            distances_sorted,
            S=kwargs['S'],
            online=kwargs['online'],
            curve=kwargs['curve'],
            interp_method=kwargs['interp_method'],
            direction=kwargs['direction'],
            polynomial_degree=kwargs['polynomial_degree'],
        )

        eps = distances_sorted[k1.knee]

    elif method == 'mean':
        eps = np.mean(distances_sorted) * kwargs['mean_multiplier']

    elif method == 'median':
        eps = np.median(distances_sorted) * kwargs['median_multiplier']

    if plot:
        fig, ax = plt.subplots(figsize=plt_size)
        ax.plot(distances_sorted)
        ax.axhline(eps, color='r', linestyle='--')
        ax.set_xlabel('Sorted Distance Index')
        ax.set_ylabel('Nearest Neighbour Distance')
        ax.set_title(f'Estimated eps: {eps:.4f}')
        plt.show()

    return eps
