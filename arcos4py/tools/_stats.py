"""Module containing tools to calculate statistics of collective events.

Example:
    >>> from arcos4py.tools import calculate_statistics
    >>> out = calculate_statistics(data = data,frame_column = "frame", collid_column = "collid")
"""

from typing import Union, List

import numpy as np
import pandas as pd
import warnings

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform


def calculate_per_frame_statistics(
    data: pd.DataFrame,
    frame_column: str,
    collid_column: str,
    obj_id_column: Union[str, None] = None,
    posCol: Union[List[str], None] = None,
) -> pd.DataFrame:
    """
    Calculate per-frame statistics for collective events.

    Parameters:
    - data (pd.DataFrame): Input data containing information on the collective events.
    - frame_column (str): The column name representing the frame numbers.
    - collid_column (str): The column name representing the collective event IDs.
    - obj_id_column (str, optional): The column name representing the object IDs. Defaults to None.
    - posCol (List[str], optional): List of column names representing the position coordinates. Defaults to None.

    Returns:
    - pd.DataFrame: A DataFrame containing the per-frame statistics.

    Output columns:
    - collid: The unique ID representing each collective event.
    - frame: The frame number.
    - current_size: The number of unique objects present in each frame of each collective event
    (calculated if obj_id_column is provided).
    - centroid_x, centroid_y: The x and y coordinates of the centroid of all objects in each frame
    (calculated if posCol is provided).
    - bounding_box_x_min, bounding_box_x_max, bounding_box_y_min, bounding_box_y_max: The minimum and maximum x and y
    coordinates forming the bounding box enclosing all objects in each frame (calculated if posCol is provided).
    - spatial_extent: The maximum pairwise distance between objects in each frame, providing a measure of the spatial
    extent of the collective event in each frame (calculated if posCol is provided).
    - convex_hull_area: The area enclosed by the convex hull formed by all objects in each frame, providing a measure
    of the spatial occupancy of the collective event in each frame (calculated if posCol is provided).
    - convex_hull_perimeter: The perimeter of the convex hull formed by all objects in each frame, providing a measure
    of the shape complexity of the collective event in each frame (calculated if posCol is provided).
    - instantaneous_velocity: The velocity of the centroid, calculated as the Euclidean distance
    between the current centroid and the centroid in the previous frame divided by the time step (assumed to be 1),
    providing a measure of the speed of the collective event in each frame (calculated if posCol is provided).
    """
    # Rename columns for easier reference
    data = data.rename(columns={frame_column: 'frame', collid_column: 'collid'})

    # Create a groupby object based on 'collid' and 'frame'
    groupby_obj = data.groupby(['collid', 'frame'])

    # Initialize an empty list to store the per-frame statistics
    stats_list = []

    # Dictionary to store the previous centroid to calculate instantaneous velocity
    previous_centroid = {}

    # Loop through each group in the groupby object
    for (collid, frame), group in groupby_obj:

        # Create a dictionary to store the statistics for the current group
        stats_dict = {'collid': collid, 'frame': frame}

        # Calculate current size (if obj_id_column is provided)
        if obj_id_column:
            stats_dict['current_size'] = group[obj_id_column].nunique()

        # Calculate other statistics (if posCol is provided)
        if posCol:

            # Calculate centroid
            centroid = group[posCol].mean().values
            for i, col in enumerate(posCol):
                stats_dict[f'centroid_{col}'] = centroid[i]

            # Calculate bounding box
            bounding_box = [group[col].min() for col in posCol] + [group[col].max() for col in posCol]
            (
                stats_dict['bounding_box_min_x'],
                stats_dict['bounding_box_min_y'],
                stats_dict['bounding_box_max_x'],
                stats_dict['bounding_box_max_y'],
            ) = bounding_box

            # Calculate spatial extent
            if len(group) > 1:
                spatial_extent = pdist(group[posCol].values).max()
            else:
                spatial_extent = 0
            stats_dict['spatial_extent'] = spatial_extent

            # Calculate convex hull area and perimeter
            if len(group) > 2:
                hull = ConvexHull(group[posCol].values)
                stats_dict['convex_hull_area'] = hull.volume
                stats_dict['convex_hull_perimeter'] = hull.area
            else:
                stats_dict['convex_hull_area'] = 0
                stats_dict['convex_hull_perimeter'] = 0

            # Calculate instantaneous velocity
            if collid in previous_centroid:
                displacement = np.linalg.norm(centroid - previous_centroid[collid])
                stats_dict['instantaneous_velocity'] = displacement
            else:
                stats_dict['instantaneous_velocity'] = 0
            previous_centroid[collid] = centroid

        # Add the statistics dictionary to the list of statistics
        stats_list.append(stats_dict)

    # Create a DataFrame from the list of statistics
    stats_df = pd.DataFrame(stats_list)

    return stats_df


def calculate_statistics(
    data: pd.DataFrame,
    frame_column: str,
    collid_column: str,
    obj_id_column: Union[str, None] = None,
    posCol: Union[List[str], None] = None,
) -> pd.DataFrame:
    """
    Calculate summary statistics for collective events based on the entire duration of each event.

    Parameters:
    - data (pd.DataFrame): Input data containing information on the collective events.
    - frame_column (str): The column name representing the frame numbers.
    - collid_column (str): The column name representing the collective event IDs.
    - obj_id_column (str, optional): The column name representing the object IDs. Defaults to None.
    - posCol (List[str], optional): List of column names representing the position coordinates. Defaults to None.

    Returns:
    - pd.DataFrame: A DataFrame containing the summary statistics of the collective events.

    Statistics Calculated:
    - collid: The unique ID representing each collective event.
    - duration: The duration of each event, calculated as the difference between the maximum and minimum frame values plus one.
    - start_frame, end_frame: The first and last frames in which each event occurs.
    - total_size: The total number of unique objects involved in each event (calculated if obj_id_column is provided).
    - min_size, max_size: The minimum and maximum size of each event, defined as the number of objects in the event's
    smallest and largest frames, respectively (calculated if obj_id_column is provided).
    - first_frame_centroid_x, first_frame_centroid_y, last_frame_centroid_x, last_frame_centroid_y:
    The x and y coordinates of the centroid of all objects in the first and last frames of each event
    (calculated if posCol is provided).
    - centroid_speed: The speed of the centroid, calculated as the distance between the first and last frame centroids
    divided by the duration (calculated if posCol is provided).
    - direction: The direction of motion of the centroid, calculated as the arctangent of the change in y divided
    the change in x (calculated if posCol is provided).
    - first_frame_bounding_box, last_frame_bounding_box: The bounding boxes enclosing all objects in the first and last
    frames of each event, defined as a list of the form `[x_min, x_max, y_min, y_max]` (calculated if posCol is provided).
    - first_frame_spatial_extent, last_frame_spatial_extent: The maximum distance between any pair of objects in the first
    and last frames (calculated if posCol is provided).
    - first_frame_convex_hull_area, last_frame_convex_hull_area: The areas of the convex hulls enclosing all objects
    in the first and last frames (calculated if posCol is provided).
    - size_variability: The standard deviation of the event size over all frames, providing a measure of the
    variability in the size of the event over time (calculated if obj_id_column is provided).
    """
    # Rename columns for easier reference
    data = data.rename(columns={frame_column: 'frame', collid_column: 'collid'})

    # Create initial stats DataFrame with collid and frame information
    stats_df = (
        data.groupby('collid')['frame']
        .agg(duration=lambda x: x.max() - x.min() + 1, start_frame='min', end_frame='max')
        .reset_index()
    )

    # If obj_id_column is provided, calculate size related stats
    if obj_id_column:
        size_stats = data.groupby('collid')[obj_id_column].agg(
            total_size='nunique',  # Get the number of unique objects
        )
        stats_df = stats_df.merge(size_stats, on='collid')

        # Calculate min and max size based on the number of objects in each frame
        frame_size_stats = data.groupby(['collid', 'frame']).size().reset_index(name='size')
        min_max_size_stats = (
            frame_size_stats.groupby('collid')['size'].agg(min_size='min', max_size='max').reset_index()
        )
        stats_df = stats_df.merge(min_max_size_stats, on='collid')

    # If posCol is provided, calculate centroid coordinates for the first and last frame of each event
    if posCol:
        centroid_data = data.groupby(['collid', 'frame'])[posCol].mean().reset_index()
        first_frame_centroid = (
            centroid_data.groupby('collid').apply(lambda x: x.loc[x['frame'].idxmin(), posCol]).reset_index()
        )
        last_frame_centroid = (
            centroid_data.groupby('collid').apply(lambda x: x.loc[x['frame'].idxmax(), posCol]).reset_index()
        )

        # Merge centroid data as separate columns
        for i, col in enumerate(posCol):
            stats_df[f'{col}_first_frame'] = first_frame_centroid[col].values
            stats_df[f'{col}_last_frame'] = last_frame_centroid[col].values

        # Calculate centroid speed and direction
        posCol_first_frame = [f'{col}_first_frame' for col in posCol]
        posCol_last_frame = [f'{col}_last_frame' for col in posCol]
        stats_df['centroid_speed'] = stats_df.apply(
            lambda row: np.linalg.norm(np.array(row[posCol_last_frame]) - np.array(row[posCol_first_frame]))
            / row['duration'],
            axis=1,
        )
        stats_df['direction'] = stats_df.apply(
            lambda row: np.arctan2(
                (row[f'{posCol[1]}_last_frame'] - row[f'{posCol[1]}_first_frame']),
                (row[f'{posCol[0]}_last_frame'] - row[f'{posCol[0]}_first_frame']),
            ),
            axis=1,
        )

        # Calculate bounding box, spatial extent, and convex hull for first and last frames
        for frame_type in ['first_frame', 'last_frame']:
            frame_data = data[
                data['frame'].isin(stats_df[f'start_frame' if frame_type == 'first_frame' else 'end_frame'])
            ]
            grouped_frame_data = frame_data.groupby('collid')[posCol].agg(['min', 'max']).reset_index()

            # Bounding box
            stats_df[
                [
                    f'bounding_box_{frame_type}_min_x',
                    f'bounding_box_{frame_type}_max_x',
                    f'bounding_box_{frame_type}_min_y',
                    f'bounding_box_{frame_type}_max_y',
                ]
            ] = grouped_frame_data.apply(
                lambda row: pd.Series(
                    [
                        row[(posCol[0], 'min')],
                        row[(posCol[0], 'max')],
                        row[(posCol[1], 'min')],
                        row[(posCol[1], 'max')],
                    ]
                ),
                axis=1,
            )

            # Spatial extent
            spatial_extent = frame_data.groupby('collid').apply(
                lambda group: pdist(group[posCol].values).max() if len(group) > 1 else 0
            )
            stats_df = stats_df.merge(
                spatial_extent.reset_index(name=f'spatial_extent_{frame_type}'), on='collid', how='left'
            )

            # Convex hull
            convex_hull_stats = frame_data.groupby('collid').apply(
                lambda group: ConvexHull(group[posCol].values).volume if len(group) > 2 else 0
            )
            stats_df = stats_df.merge(
                convex_hull_stats.reset_index(name=f'convex_hull_area_{frame_type}'), on='collid', how='left'
            )

        # Calculate size variability
        size_variability = frame_size_stats.groupby('collid')['size'].std().reset_index(name='size_variability')
        stats_df = stats_df.merge(size_variability, on='collid', how='left')

    return stats_df


class calcCollevStats:
    """Class to calculate statistics of collective events."""

    def __init__(self) -> None:
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
        posCol: Union[list, None],
    ) -> pd.DataFrame:
        warnings.warn(
            "The 'calculate' method is deprecated and will be removed in a future version. "
            "Please use the 'calculate_statistics' function instead.",
            DeprecationWarning,
        )
        return calculate_statistics(data, frame_column, collid_column, obj_id_column, posCol)
