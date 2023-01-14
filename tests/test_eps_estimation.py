import pandas as pd
import pytest

from arcos4py.tools import estimate_eps


@pytest.fixture
def test_data():
    data = {'t': [1, 1, 2, 2, 3, 3], 'x': [0, 1, 2, 3, 4, 5], 'y': [1, 2, 3, 4, 5, 6]}
    return pd.DataFrame(data)


def test_estimate_eps(test_data):
    # Test kneepoint method
    eps = estimate_eps(
        test_data,
        method='kneepoint',
        pos_cols=['x', 'y'],
        frame_col='t',
        n_neighbors=2,
        S=1,
        online=True,
        curve='convex',
        interp_method='polynomial',
        direction='increasing',
        polynomial_degree=7,
        plot=False,
    )
    assert eps is not None

    # Test mean method
    eps = estimate_eps(
        test_data, method='mean', pos_cols=['x', 'y'], frame_col='t', n_neighbors=2, mean_multiplier=1.5, plot=False
    )
    assert eps is not None

    # Test median method
    eps = estimate_eps(
        test_data, method='median', pos_cols=['x', 'y'], frame_col='t', n_neighbors=2, median_multiplier=1.5, plot=False
    )
    assert eps is not None

    # Test invalid method
    with pytest.raises(ValueError):
        estimate_eps(test_data, method='invalid', pos_cols=['x', 'y'], frame_col='t', n_neighbors=2, plot=False)

    # Test invalid column name
    with pytest.raises(ValueError):
        estimate_eps(test_data, pos_cols=['x', 'y', 'z'], frame_col='t', n_neighbors=2, plot=False)

    # Test invalid kwarg type
    with pytest.raises(ValueError):
        estimate_eps(
            test_data,
            method='mean',
            pos_cols=['x', 'y'],
            frame_col='t',
            n_neighbors=2,
            mean_multiplier='string',
            plot=False,
        )
