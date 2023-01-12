"""Tools for resampling data and bootstrapping."""

from __future__ import annotations
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from typing import Union


def _get_xy_change(
    df: pd.DataFrame, object_id_name: str = 'track_id', posCols: list = ['x', 'y']
) -> tuple(pd.DataFrame, list[str]):
    """Calculate xy change for each object."""
    # get xy change for each object
    df_new = df.copy(deep=True)
    change_cols = [f'{i}_change' for i in posCols]
    cumsum_cols = [f'{i}_cumsum_change' for i in change_cols]
    df_new[change_cols] = df_new.groupby(object_id_name)[posCols].diff(axis=0)
    df_new[cumsum_cols] = df_new.groupby(object_id_name)[change_cols].cumsum()
    df_new[cumsum_cols] = df_new[cumsum_cols].fillna(0)
    return df_new, cumsum_cols


def shuffle_tracks(
    df: pd.DataFrame, object_id_name: str = 'track_id', posCols: list = ['x', 'y'], seed=42
) -> pd.DataFrame:
    """Resample tracks by switching the first timepoint\
        positions of two tracks and then propagating the cummulative difference."""
    df_new = df.copy(deep=True)
    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique track IDs
    track_ids = df_new[object_id_name].unique()
    # Keep track of the track IDs that have been switched
    switched_tracks = set()
    # calculate relative coordinate change
    df_new, change_cols = _get_xy_change(df_new, object_id_name, posCols)
    # Iterate over the unique track IDs
    for track_id in track_ids:
        # Skip the track if it has already been switched
        if track_id in switched_tracks:
            continue
        # Get a random track ID that has not been switched yet
        available_track_ids = np.array([tid for tid in track_ids if tid not in switched_tracks and tid != track_id])
        if available_track_ids.size == 0:
            continue
        random_track_id = rng.choice(available_track_ids)
        # Get the rows with the current track ID
        track_rows = df_new[df_new[object_id_name] == track_id].copy()
        # Get the rows with the random track ID
        random_track_rows = df_new[df_new[object_id_name] == random_track_id].copy()

        # Switch the first timepoint positions of the two tracks
        df_new.loc[track_rows.index, posCols] = random_track_rows[posCols].iloc[0].to_numpy()
        df_new.loc[random_track_rows.index, posCols] = track_rows[posCols].iloc[0].to_numpy()
        # Add the relative position shift to the rest of the timepoints for the current track
        df_new.loc[track_rows.index, posCols] += track_rows[change_cols].to_numpy()
        # Add the relative position shift to the rest of the timepoints for the random track
        df_new.loc[random_track_rows.index, posCols] += random_track_rows[change_cols].to_numpy()
        # Add the switched track IDs to the set
        switched_tracks.add(track_id)
        switched_tracks.add(random_track_id)
    return df_new


def shuffle_timepoints(
    df: pd.DataFrame,
    objet_id_name: str = 'track_id',
    frame_column: str = 'time',
    seed=42,
) -> pd.DataFrame:
    """Resample data by shuffling data from timepoints on a per trajectory basis."""
    df_new = df.copy(deep=True)
    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique track IDs
    track_ids = df_new[objet_id_name].unique()
    # Iterate over the unique track IDs
    for track_id in track_ids:
        # Get the rows with the current track ID
        track_rows = df_new[df_new[objet_id_name] == track_id].copy()
        # Shuffle the timepoints
        track_rows_np = track_rows[frame_column].to_numpy()
        rng.shuffle(track_rows_np)
        # Set the shuffled timepoints
        df_new.loc[track_rows.index, frame_column] = track_rows_np
    df_new.sort_values(by=[frame_column], inplace=True)
    return df_new


def _get_activity_blocks(data: np.ndarray) -> list[np.ndarray]:
    """Get the activity blocks of a binary activity column."""
    # Get the indices of the activity blocks
    activity_block_indices = np.where(np.diff(data) != 0)[0] + 1
    # Get the activity blocks
    activity_blocks = np.split(data, activity_block_indices)
    # Remove empty activity blocks
    activity_blocks = [block for block in activity_blocks if block.size > 0]
    return activity_blocks


def shuffle_activity_bocks_per_trajectory(
    df: pd.DataFrame, objet_id_name: str, frame_column: str, meas_column: str, seed=42
) -> pd.DataFrame:
    """Resample data by shuffling the activity blocks of a binary activity column on a per trajectory basis."""
    df_new = df.copy(deep=True)
    # check if data in meas_column is binary
    if not np.array_equal(np.unique(df_new[meas_column]), np.array([0, 1])):
        raise ValueError('Data in meas_column must be binary')

    # raise warning if 1 makes up more than 25% of the array
    if np.sum(df_new[meas_column]) / df_new[meas_column].size > 0.25:
        warnings.warn(
            'More than 25%% of the data in meas_column is 1.\
            This could impact the validity of this resampling approach.'
        )

    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique track IDs
    track_ids = df_new[objet_id_name].unique()
    # Iterate over the unique track IDs
    for track_id in track_ids:
        # Get the rows with the current track ID
        track_rows = df_new[df_new[objet_id_name] == track_id].copy()
        # Get the activity blocks
        meas_array = track_rows[meas_column].to_numpy()
        activity_blocks = _get_activity_blocks(meas_array)
        # Shuffle the activity blocks
        rng.shuffle(activity_blocks)
        # Set the shuffled activity blocks
        df_new.loc[track_rows.index, meas_column] = np.concatenate(activity_blocks)
    df_new.sort_values(by=[frame_column], inplace=True)
    return df_new


def shuffle_coordinates_per_timepoint(df: pd.DataFrame, posCols: list[str], frame_column: str, seed=42) -> pd.DataFrame:
    """Resample data by shuffling the coordinates of a trajectory on a per timepoint basis."""
    df_new = df.copy(deep=True)
    # Get unique timepoints
    timepoints = df_new[frame_column].unique()

    # Set the random seed
    rng = np.random.default_rng(seed)
    # Iterate over the unique timepoints
    for timepoint in timepoints:
        # Get the rows with the current timepoint
        timepoint_rows = df_new[df_new[frame_column] == timepoint].copy()
        # Shuffle the coordinates
        timepoint_rows_np = timepoint_rows[posCols].to_numpy()
        rng.shuffle(timepoint_rows_np)
        # Set the shuffled coordinates
        df_new.loc[timepoint_rows.index, posCols] = timepoint_rows_np
    return df_new


def shift_timepoints_per_trajectory(df: pd.DataFrame, objet_id_name: str, frame_column: str, seed=42) -> pd.DataFrame:
    """Resample data by shifting the timepoints a random ammount of a trajectory on a per trajectory basis."""
    df_new = df.copy(deep=True)
    # Get unique track IDs
    track_ids = df_new[objet_id_name].unique()

    # Set the random seed
    rng = np.random.default_rng(seed)
    # Iterate over the unique track IDs
    for track_id in track_ids:
        # Get the rows with the current track ID
        track_rows = df_new[df_new[objet_id_name] == track_id].copy()
        # Get the timepoints
        timepoints = track_rows[frame_column].to_numpy()
        # Get the shift
        shift = rng.integers(-timepoints.size, timepoints.size)
        # Shift the timepoints
        timepoints = np.roll(timepoints, shift)
        # Set the shifted timepoints
        df_new.loc[track_rows.index, frame_column] = timepoints
    df_new.sort_values(by=[frame_column], inplace=True)
    return df_new


def resample_data(
    data: pd.DataFrame,
    posCols: list,
    frame_column: str,
    id_column: str,
    meas_column: Union[str, None] = None,
    method: str = 'shuffle_tracks',
    n=100,
    seed=42,
    show_progress=True,
    verbose=False,
) -> pd.DataFrame:
    """Resamples data in order to perform bootstrapping analysis.

    Arguments:
        data (pd.Dataframe): The data to bootstrap
        posCols (list, optional): The columns to use for the position.
        frame_column (str, optional): The column to use for the frame.
        id_column (str, optional): The column to use for the object ID.
        meas_column (str, optional): The column to use for the measurement.
            Only needed for 'activity_blocks_shuffle'. Defaults to 'm'.
        method (str, optional): The method to use for bootstrapping. Defaults to 'shuffle_tracks'.
            Available methods are: "shuffle_tracks", 'shuffle_tracks': shuffle_track, 'shuffle_timepoints',
            'shift_timepoints', 'shuffle_binary_blocks', 'shuffle_coordinates_timepoint'
        n (int, optional): The number of resample iterations. Defaults to 100.
        seed (int, optional): The random seed. Defaults to 42.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        pd.DataFrame: The bootstrapped data
    """
    method_dict: dict[str, callable] = {
        'shuffle_tracks': shuffle_tracks,
        'shuffle_timepoints': shuffle_timepoints,
        'shift_timepoints': shift_timepoints_per_trajectory,
        'shuffle_binary_blocks': shuffle_activity_bocks_per_trajectory,
        'shuffle_coordinates_timepoint': shuffle_coordinates_per_timepoint,
    }

    function_args: dict[str, tuple] = {
        'shuffle_tracks': (id_column, posCols),
        'shuffle_timepoints': (id_column, frame_column),
        'shift_timepoints': (id_column, frame_column),
        'shuffle_binary_blocks': (id_column, frame_column, meas_column),
        'shuffle_coordinates_timepoint': (posCols, frame_column),
    }

    if method not in method_dict.keys():
        raise ValueError(f'method must be one of {method_dict.keys()}')
    if method == 'shuffle_binary_blocks' and meas_column is None:
        raise ValueError('meas_column must be set for binary_blocks_shuffle')

    # Check if the columns are in the data
    if method == 'shuffle_binary_blocks':
        relevant_columns = posCols + [frame_column, id_column, meas_column]
    else:
        relevant_columns = posCols + [frame_column, id_column]

    # Check if the columns are in the data
    for i in relevant_columns:
        if i not in data.columns:
            raise ValueError(f'{i} not in df.columns')

    # check if there are any Nan in the columns selected
    na_cols = []
    for i in relevant_columns:
        if data[posCols].isnull().values.any():
            na_cols.append(i)
    if na_cols:
        warnings.warn(f'NaN values in {na_cols}, default behaviour is to drop these rows')
        data.dropna(subset=na_cols, inplace=True)

    # Check if data is sorted
    if not data[frame_column].is_monotonic_increasing:
        # Sort the data
        data = data.sort_values(frame_column)

    rng = np.random.default_rng(seed)
    # create a list of random numbers between 0 and 1000000
    seed_list = rng.integers(1_000_000_000, size=n)
    df_out = []
    # shuffle xy position for each object
    if verbose:
        print(f'Resampling for each object {n} times')

    bootstrapping_func = method_dict[method]
    for i in tqdm(range(n), disable=not show_progress):
        _seed = seed_list[i]
        data_new = bootstrapping_func(data, *function_args[method], seed=_seed)
        if verbose:
            print(f'Original data: {data}')
            print(f'Shuffled data: {data_new}')

        data_new['iteration'] = np.repeat(i, len(data_new))
        df_out.append(data_new)
    return pd.concat(df_out)
