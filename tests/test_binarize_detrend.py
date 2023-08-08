import numpy as np
import pandas as pd
import pytest

from arcos4py.tools._binarize_detrend import binData, detrender


def test_detrender_init():
    det = detrender()
    assert det.smoothK == 3
    assert det.biasK == 51
    assert det.peakThr == 0.2
    assert det.polyDeg == 1
    assert det.biasMet == "runmed"
    assert det.n_jobs == 1

    with pytest.raises(ValueError):
        detrender(biasMet="invalid_method")


def test_detrender_runmed():
    x = np.array([1, 2, 3, 4, 5])
    det = detrender()
    result = det._detrend_runnmed(x, filter_size=3, endrule_mode="nearest")
    assert result.size == x.size


def test_binData_init():
    binarizer = binData()
    assert binarizer.smoothK == 3
    assert binarizer.biasK == 51
    assert binarizer.peakThr == 0.2
    assert binarizer.binThr == 0.1
    assert binarizer.polyDeg == 1
    assert binarizer.biasMet == "runmed"
    assert binarizer.n_jobs == 1


def test_binData_run_without_groupcol():
    binarizer = binData(biasMet="none")
    data = pd.DataFrame({'ERK_KTR': [1, 2, 3], 'Frame': [1, 2, 3]})
    result = binarizer.run(data, colGroup=None, colMeas='ERK_KTR', colFrame='Frame')
    assert 'ERK_KTR.resc' in result.columns
    assert 'ERK_KTR.bin' in result.columns


def test_binData_run_with_groupcol():
    binarizer = binData()
    data = pd.DataFrame({'trackID': [1, 1, 2], 'ERK_KTR': [1, 2, 3], 'Frame': [1, 2, 3]})
    result = binarizer.run(data, colGroup='trackID', colMeas='ERK_KTR', colFrame='Frame')
    assert 'ERK_KTR.resc' in result.columns
    assert 'ERK_KTR.bin' in result.columns


def test_median_detrending():
    df = pd.read_csv('tests/testdata/peak_data_linear_trend.csv')
    df_test = pd.read_csv('tests/testdata/peak_data_linear_trend_out.csv')
    binarizer = binData(smoothK=1, biasK=15, binThr=0.3, biasMet="runmed")
    df_out = binarizer.run(df, colGroup='trackID', colMeas='ERK_KTR', colFrame='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_polynomial_detrending():
    df = pd.read_csv('tests/testdata/peak_data_lm.csv')
    df_test = pd.read_csv('tests/testdata/peak_data_lm_out.csv')
    binarizer = binData(smoothK=1, binThr=0.3, biasMet="lm")
    df_out = binarizer.run(df, colGroup='trackID', colMeas='ERK_KTR', colFrame='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_no_detrending():
    df = pd.read_csv('tests/testdata/peak_data.csv')
    df_test = pd.read_csv('tests/testdata/peak_data_out.csv')
    binarizer = binData(smoothK=1, binThr=0.6, biasMet="none")
    df_out = binarizer.run(df, colGroup='trackID', colMeas='ERK_KTR', colFrame='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_no_detrending_without_groupcol():
    df = pd.read_csv('tests/testdata/peak_data.csv')
    df = df.query('trackID == 1')
    df_test = pd.read_csv('tests/testdata/peak_data_out.csv')
    df_test = df_test.query('trackID == 1')

    binarizer = binData(smoothK=1, binThr=0.6, biasMet="none")
    df_out = binarizer.run(df, colGroup=None, colMeas='ERK_KTR', colFrame='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_detrending_with_groupcol():
    df = pd.read_csv('tests/testdata/peak_data.csv')
    binarizer = binData(smoothK=1, binThr=0.6, biasMet="runmed")

    with pytest.warns(
        UserWarning, match="No detrending is performed, only rescaling. To run detrending, set colGroup."
    ):
        binarizer.run(df, colGroup=None, colMeas='ERK_KTR', colFrame='Frame')
