from numpy import NaN
import pytest
from arcos_py.tools import interpolation
import pandas as pd

from pandas.testing import assert_frame_equal


@pytest.fixture
def test_data():
    d = {
        't': [i if i != 4 else NaN for i in range(1, 8)],
        'y': [i for i in range(1, 8)],
        'id': [1 for i in range(7)],
    }
    df = pd.DataFrame(d)

    d2 = {
        't': [i for i in range(1, 8)],
        'y': [i for i in range(1, 8)],
        'id': [1 for i in range(7)],
    }
    df2 = pd.DataFrame(d2)
    return df, df2


def test_interpolate_middle(test_data: pd.DataFrame):
    df, df_out = test_data
    df['t'] = interpolation(df['t'].to_numpy()).interpolate()
    df['t'] = df['t'].astype(int)
    assert_frame_equal(df, df_out)
