"""Module containing tools to calculate statistics of collective events.

Example:
    >>> from arcos4py.tools import calculate_statistics
    >>> out = calculate_statistics(data = data,frame_column = "frame", collid_column = "collid")
"""

import warnings
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def calculate_statistics_per_frame(
    data: pd.DataFrame,
    frame_column: str,
    collid_column: str,
    pos_columns: Union[List[str], None] = None,
) -> pd.DataFrame:
    """Calculate summary statistics for collective events based on the entire duration of each event.

    Arguments:
        data (pd.DataFrame): Input data containing information on the collective events.
        frame_column (str): The column name representing the frame numbers.
        collid_column (str): The column name representing the collective event IDs.
        pos_columns (List[str], optional): List of column names representing the position coordinates. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics of the collective events.

    Statistics Calculated:
        - collid: The unique ID representing each collective event.
        - frame: The frame number.
        - size: The number of objects in the collective event
        - centroid_x, centroid_y: The x and y coordinates of the centroid of all objects in the collective event
            (calculated if pos_columns is provided).
        - spatial_extent: The maximum distance between any pair of objects in the collective event
            (calculated if pos_columns is provided).
        - convex_hull_area: The area of the convex hull enclosing all objects in the collective event
            (calculated if pos_columns is provided).
        - direction: The direction of motion of the centroid, calculated as the arctangent of the change in y divided
            the change in x (calculated if pos_columns is provided).
        - centroid_speed: The speed of the centroid, calculated as the norm of the change
            in x and y divided by the duration (calculated if pos_columns is provided).
    """
    necessary_columns = [frame_column, collid_column]
    if pos_columns:
        necessary_columns.extend(pos_columns)

    for col in necessary_columns:
        if col not in data.columns and col is not None:
            raise ValueError(f"The column '{col}' is not present in the input data.")

    data = data.rename(columns={frame_column: 'frame', collid_column: 'collid'})
    collid_groups = data.groupby(['frame', 'collid'])
    stats_list = []

    for (frame, collid), group_data in collid_groups:

        frame_stats = {'collid': collid, 'frame': frame}

        frame_stats['size'] = group_data.count()['frame']

        # If pos_columns are provided, calculate spatial statistics for this frame
        if pos_columns:
            # Calculate centroid
            centroid = group_data[pos_columns].mean().to_dict()
            for pos_col, cent_val in centroid.items():
                frame_stats[f'centroid_{pos_col}'] = cent_val

            # Calculate spatial extent
            spatial_extent = pdist(group_data[pos_columns].values).max() if len(group_data) > 1 else 0
            frame_stats['spatial_extent'] = spatial_extent

            # Calculate convex hull area
            convex_hull_area = (
                ConvexHull(group_data[pos_columns].values).volume if len(group_data) > len(pos_columns) else 0
            )
            frame_stats['convex_hull_area'] = convex_hull_area

        stats_list.append(frame_stats)

    # Create a DataFrame from the list of statistics
    stats_df = pd.DataFrame(stats_list)

    # If pos_columns are provided, we can calculate speed and direction by looking at changes between frames
    if pos_columns:
        stats_df.sort_values(by=['collid', 'frame'], inplace=True)

        for i, col in enumerate(pos_columns):
            stats_df[f'delta_{col}'] = stats_df.groupby('collid')[f'centroid_{col}'].diff()

        # Calculate speed (the norm of the delta vector)
        stats_df['centroid_speed'] = np.linalg.norm(stats_df[[f'delta_{col}' for col in pos_columns]].values, axis=1)

        # Calculate direction (only for 2D)
        if len(pos_columns) == 2:
            stats_df['direction'] = np.arctan2(stats_df['delta_' + pos_columns[1]], stats_df['delta_' + pos_columns[0]])

        # Clean up temporary delta columns
        stats_df.drop(columns=[f'delta_{col}' for col in pos_columns], inplace=True)

    return stats_df


def calculate_statistics(
    data: pd.DataFrame,
    frame_column: str,
    collid_column: str,
    obj_id_column: Union[str, None] = None,
    pos_columns: Union[List[str], None] = None,
) -> pd.DataFrame:
    """Calculate summary statistics for collective events based on the entire duration of each event.

    Arguments:
        data (pd.DataFrame): Input data containing information on the collective events.
        frame_column (str): The column name representing the frame numbers.
        collid_column (str): The column name representing the collective event IDs.
        obj_id_column (str, optional): The column name representing the object IDs. Defaults to None.
        pos_columns (List[str], optional): List of column names representing the position coordinates. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics of the collective events.

    Statistics Calculated:
        - collid: The unique ID representing each collective event.
        - duration: The duration of each event, calculated as the difference between the maximum
            and minimum frame values plus one.
        - first_timepoint, last_timepoint: The first and last frames in which each event occurs.
        - total_size: The total number of unique objects involved in each event
            (calculated if obj_id_column is provided).
        - min_size, max_size: The minimum and maximum size of each event,
            defined as the number of objects in the event's smallest and largest frames, respectively.
        - first_frame_centroid_x, first_frame_centroid_y, last_frame_centroid_x, last_frame_centroid_y:
            The x and y coordinates of the centroid of all objects in the first and last frames of each event
            (calculated if posCol is provided).
        - centroid_speed: The speed of the centroid, calculated as the distance between
            the first and last frame centroids divided by the duration (calculated if posCol is provided).
        - direction: The direction of motion of the centroid, calculated as the arctangent of the change in y divided
            the change in x (calculated if posCol is provided).
        - first_frame_spatial_extent, last_frame_spatial_extent: The maximum distance between any pair of objects in the
        first and last frames (calculated if posCol is provided).
        - first_frame_convex_hull_area, last_frame_convex_hull_area: The areas of the convex hulls enclosing all objects
            in the first and last frames (calculated if posCol is provided).
        - size_variability: The standard deviation of the event size over all frames, providing a measure of the
            variability in the size of the event over time (calculated if obj_id_column is provided).
    """
    # Error handling: Check if necessary columns are present in the input data
    necessary_columns = [frame_column, collid_column]
    if obj_id_column:
        necessary_columns.append(obj_id_column)
    if pos_columns:
        necessary_columns.extend(pos_columns)

    for col in necessary_columns:
        if col not in data.columns and col is not None:
            raise ValueError(f"The column '{col}' is not present in the input data.")

    # Rename columns for easier reference
    data = data.rename(columns={frame_column: 'frame', collid_column: 'collid'})

    collid_groups = data.groupby('collid')

    # Initialize an empty list to store the statistics
    stats_list = []

    for collid, group_data in collid_groups:

        collid_stats = {'collid': collid}

        # Grouping by 'collid' to get initial statistics
        duration = group_data['frame'].max() - group_data['frame'].min() + 1
        collid_stats['duration'] = duration
        collid_stats['first_timepoint'] = group_data['frame'].min()
        collid_stats['last_timepoint'] = group_data['frame'].max()

        # If obj_id_column is provided, calculate size related stats
        if obj_id_column:
            total_size = group_data[obj_id_column].nunique()

            collid_stats['total_size'] = total_size

        # calculate min and max size based on the number of objects in each frame
        frame_size_stats = group_data.groupby('frame').size()
        collid_stats['min_size'] = frame_size_stats.min()
        collid_stats['max_size'] = frame_size_stats.max()

        # If posCol is provided, calculate centroid coordinates for the
        if pos_columns:
            tp_1 = collid_stats['first_timepoint']
            tp_2 = collid_stats['last_timepoint']

            centroid_data = group_data.groupby('frame')[pos_columns].mean().reset_index()

            for col in pos_columns:
                collid_stats[f'first_frame_centroid_{col}'] = centroid_data.query(f'frame == {tp_1}')[col].to_numpy()[0]
                collid_stats[f'last_frame_centroid_{col}'] = centroid_data.query(f'frame == {tp_2}')[col].to_numpy()[0]

            # Calculate speed and direction
            speed = np.linalg.norm(
                np.column_stack([collid_stats[f'first_frame_centroid_{col}'] for col in pos_columns])
                - np.column_stack([collid_stats[f'last_frame_centroid_{col}'] for col in pos_columns]),
                axis=1,
            ) / (collid_stats['duration'] - 1)

            collid_stats['centroid_speed'] = speed[0]

            # Direction For 2D data
            if len(pos_columns) == 2:
                collid_stats['direction'] = np.arctan2(
                    collid_stats[f'last_frame_centroid_{pos_columns[1]}']
                    - collid_stats[f'first_frame_centroid_{pos_columns[1]}'],
                    collid_stats[f'last_frame_centroid_{pos_columns[0]}']
                    - collid_stats[f'first_frame_centroid_{pos_columns[0]}'],
                )
            # Direction For 3D data
            elif len(pos_columns) == 3:
                dx = (
                    collid_stats[f'last_frame_centroid_{pos_columns[0]}']
                    - collid_stats[f'first_frame_centroid_{pos_columns[0]}']
                )
                dy = (
                    collid_stats[f'last_frame_centroid_{pos_columns[1]}']
                    - collid_stats[f'first_frame_centroid_{pos_columns[1]}']
                )
                dz = (
                    collid_stats[f'last_frame_centroid_{pos_columns[2]}']
                    - collid_stats[f'first_frame_centroid_{pos_columns[2]}']
                )

                # Calculate azimuth and elevation
                collid_stats['azimuth'] = np.arctan2(dy, dx)
                collid_stats['elevation'] = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
            else:
                raise ValueError("Position columns can only be 2 or 3.")

            # Loop over first and last frames separately to calculate the spatial extent and convex hull area
            for frame_name, frame_number in zip(['first_frame', 'last_frame'], [tp_1, tp_2]):
                # Get data for either the first or last frame
                frame_data = group_data.query(f'frame == {frame_number}')

                # Calculate spatial extent
                spatial_extent = pdist(frame_data[pos_columns].values).max() if len(frame_data) > 1 else 0
                collid_stats[f'{frame_name}_spatial_extent'] = spatial_extent

                # Calculate convex hull area
                convex_hull_area = (
                    ConvexHull(frame_data[pos_columns].values).volume if len(frame_data) > len(pos_columns) else 0
                )
                collid_stats[f'{frame_name}_convex_hull_area'] = convex_hull_area

        stats_list.append(collid_stats)

    # Create a DataFrame from the list of statistics
    stats_df = pd.DataFrame(stats_list)

    # Calculate size variability
    if obj_id_column:
        # Calculating size for each collid and frame
        frame_size_stats = data.groupby(['collid', 'frame'])[obj_id_column].nunique().reset_index(name='size')
        size_variability = frame_size_stats.groupby('collid')['size'].std().reset_index(name='size_variability')
        stats_df = stats_df.merge(size_variability, on='collid', how='left')

    return stats_df


class calcCollevStats:
    """Class to calculate statistics of collective events."""

    def __init__(self) -> None:
        """Initialize the class."""
        warnings.warn(
            "The 'calcCollevStats' class is deprecated and will be removed in a future version. "
            "Please use the standalone functions instead (calculate_statistics).",
            DeprecationWarning,
        )

    def calculate(
        self,
        data: pd.DataFrame,
        frame_column: str,
        collid_column: str,
        obj_id_column: Union[str, None],
        posCol: Union[list, None] = None,
    ) -> pd.DataFrame:
        """Calculate summary statistics for collective events based on the entire duration of each event.

        Arguments:
            data (pd.DataFrame): Input data containing information on the collective events.
            frame_column (str): The column name representing the frame numbers.
            collid_column (str): The column name representing the collective event IDs.
            obj_id_column (str, optional): The column name representing the object IDs. Defaults to None.
            posCol (list, optional): List of column names representing the position coordinates. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the summary statistics of the collective events.

        Deprecated:
            calculate: Use calculate_statistics instead.
        """
        warnings.warn(
            "The 'calculate' method is deprecated and will be removed in a future version. "
            "Please use the 'calculate_statistics' function instead.",
            DeprecationWarning,
        )
        return calculate_statistics(data, frame_column, collid_column, obj_id_column, posCol)
