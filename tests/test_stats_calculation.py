import pandas as pd
import numpy as np

import pytest
from arcos4py.tools import calculate_statistics



@pytest.fixture
def test_data():
    data = pd.DataFrame({
        'collid': [1, 1, 1, 2, 2, 2],
        'frame': [1, 2, 3, 1, 2, 3],
        'id': [1, 2, 3, 4, 5, 6],
        'x': [0, 1, 2, 3, 4, 5],
        'y': [0, 1, 2, 3, 4, 5],
    })
    return data


def test_calculate_summary_statistics(test_data):
    data = test_data
    # Test with obj_id_column and posCol
    stats_df = calculate_statistics(data, 'frame','collid', 'id', ['x', 'y'])

    # Expected output
    expected_output = pd.DataFrame({
        'collid': [1, 2],
        'duration': [3, 3],
        'first_timepoint': [1, 1],
        'last_timepoint': [3, 3],
        'total_size': [3, 3],
        'min_size': [1, 1],
        'max_size': [1, 1],
        'first_frame_centroid_x': [0, 3],
        'last_frame_centroid_x': [2, 5],
        'first_frame_centroid_y': [0, 3],
        'last_frame_centroid_y': [2, 5],
        'centroid_speed': [2*np.sqrt(2)/2, 2*np.sqrt(2)/2],
        'direction': [np.pi/4, np.pi/4],
        'first_frame_spatial_extent': [0, 0],
        'first_frame_convex_hull_area': [0, 0],
        'last_frame_spatial_extent': [0, 0],
        'last_frame_convex_hull_area': [0, 0],
        'size_variability': [0, 0],
    })

    pd.testing.assert_frame_equal(stats_df, expected_output, check_dtype=False)

    # Test with obj_id_column only
    stats_df = calculate_statistics(data, 'frame', 'collid', 'id', None)

    # Expected output
    expected_output = pd.DataFrame({
        'collid': [1, 2],
        'duration': [3, 3],
        'first_timepoint': [1, 1],
        'last_timepoint': [3, 3],
        'total_size': [3, 3],
        'min_size': [1, 1],
        'max_size': [1, 1],
        'size_variability': [0.0, 0.0],
    })

    pd.testing.assert_frame_equal(stats_df, expected_output)

    # Test with posCol only
    stats_df = calculate_statistics(data, 'frame', 'collid', None, ['x', 'y'])

    # Expected output
    expected_output = pd.DataFrame({
        'collid': [1, 2],
        'duration': [3, 3],
        'first_timepoint': [1, 1],
        'last_timepoint': [3, 3],
        'min_size': [1, 1],
        'max_size': [1, 1],
        'first_frame_centroid_x': [0.0, 3.0],
        'last_frame_centroid_x': [2, 5.0],
        'first_frame_centroid_y': [0, 3],
        'last_frame_centroid_y': [2, 5],
        'centroid_speed': [2*np.sqrt(2)/2, 2*np.sqrt(2)/2],
        'direction': [np.pi/4, np.pi/4],
        'first_frame_spatial_extent': [0.0, 0.0],
        'first_frame_convex_hull_area': [0.0, 0.0],
        'last_frame_spatial_extent': [0.0, 0.0],
        'last_frame_convex_hull_area': [0.0, 0.0],
    })

    pd.testing.assert_frame_equal(stats_df, expected_output, check_dtype=False)