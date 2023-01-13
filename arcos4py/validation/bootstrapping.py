"""Tools for resampling data and bootstrapping."""

from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from arcos4py import ARCOS
from arcos4py.tools import calcCollevStats

from .resampling import resample_data


def bootstrap_arcos(
    df: pd.DataFrame,
    posCols: list,
    frame_column: str,
    id_column: str,
    meas_column: Union[str, None] = None,
    method: Union[str, list[str]] = 'shuffle_tracks',
    smoothK: int = 3,
    biasK: int = 51,
    peakThr: float = 0.2,
    binThr: float = 0.1,
    polyDeg: int = 1,
    biasMet: str = "runmed",
    eps: float = 2,
    epsPrev: float | None = None,
    minClsz: int = 1,
    nPrev: int = 1,
    stats_metric: str = "total_size",
    n=100,
    seed=42,
    show_progress=True,
    verbose=False,
    paralell_processing=True,
) -> pd.DataFrame:
    """Bootstrap data using the ARCOS algorithm.

    Arugments:
        df: DataFrame containing the data to be bootstrapped.
        posCols: List of column names containing the x and y coordinates.
        frame_column: Name of the column containing the frame number.
        id_column: Name of the column containing the track id.
        meas_column: Name of the column containing the measurement.
        method: Method used for bootstrapping. Can be either 'shuffle_tracks' or 'shift_timepoints'.
        n: Number of bootstraps.
        seed: Seed for the random number generator.
        show_progress: Show a progress bar.
        verbose: Print additional information.

    Returns:
        DataFrame containing the bootstrapped data.
    """
    if stats_metric not in [
        "duration",
        "total_size",
        "min_size",
        "max_size",
        "start_frame",
        "end_frame",
        "first_frame_centroid",
        "last_frame_centroid",
    ]:
        raise ValueError(f"Invalid metric: {stats_metric}")

    clid_name = 'clid'
    stats_df_list = []

    df_resampled = resample_data(
        data=df,
        posCols=posCols,
        frame_column=frame_column,
        id_column=id_column,
        meas_column=meas_column,
        method=method,
        n=n,
        seed=seed,
        show_progress=show_progress,
        verbose=verbose,
        paralell_processing=paralell_processing,
    )

    iterations = df_resampled['iteration'].unique()

    for i_iter in tqdm(iterations, disable=not show_progress):
        stats_df = _apply_arcos(
            i_iter=i_iter,
            df_resampled=df_resampled,
            posCols=posCols,
            frame_column=frame_column,
            id_column=id_column,
            meas_column=meas_column,
            smoothK=smoothK,
            biasK=biasK,
            peakThr=peakThr,
            binThr=binThr,
            polyDeg=polyDeg,
            biasMet=biasMet,
            eps=eps,
            epsPrev=epsPrev,
            minClsz=minClsz,
            nPrev=nPrev,
            clid_name=clid_name,
        )
        stats_df_list.append(stats_df)

    stats_df = pd.concat(stats_df_list, ignore_index=True)

    stats_df_mean = (
        stats_df[['bootstrap_iteration', stats_metric]].groupby(['bootstrap_iteration']).agg(['mean']).reset_index()
    )
    # for bootstrap iteratoins that did not detect any events, set the metric to 0
    stats_df_mean[stats_metric] = stats_df_mean[stats_metric].fillna(0)
    stats_df_mean.plot.hist(x='bootstrap_iteration', y=stats_metric, bins=len(stats_df_mean))
    it_0_value = stats_df_mean[stats_df_mean['bootstrap_iteration'] == 0]['total_size']['mean'].to_numpy()[0]
    plt.axvline(it_0_value, color='red', ls='--')
    mean_0 = stats_df_mean[stats_metric]['mean'][0]
    means = stats_df_mean[stats_metric]['mean'][1:]
    p_value = sum(means > mean_0) / len(means)
    return stats_df, p_value


def _apply_arcos(
    i_iter: int,
    df_resampled: pd.DataFrame,
    posCols: list[str],
    frame_column: str,
    id_column: str,
    meas_column: str,
    smoothK: int,
    biasK: int,
    peakThr: float,
    binThr: float,
    polyDeg: int,
    biasMet: str,
    eps: float,
    epsPrev: float,
    minClsz: int,
    nPrev: int,
    clid_name: str,
) -> pd.DataFrame:
    df_i_iter = df_resampled.loc[df_resampled['iteration'] == i_iter]
    arcos_instance = ARCOS(
        data=df_i_iter,
        posCols=posCols,
        frame_column=frame_column,
        id_column=id_column,
        measurement_column=meas_column,
        clid_column=clid_name,
    )
    arcos_instance.interpolate_measurements()
    arcos_instance.bin_measurements(
        smoothK=smoothK, biasK=biasK, peakThr=peakThr, binThr=binThr, polyDeg=polyDeg, biasMet=biasMet
    )
    df_arcos = arcos_instance.trackCollev(eps=eps, epsPrev=epsPrev, minClsz=minClsz, nPrev=nPrev)

    if i_iter == 0 and df_arcos.empty:
        raise ValueError('No events detected in control, consider changing parameters')

    stats_df = calcCollevStats().calculate(df_arcos, frame_column, clid_name, id_column, posCols)
    stats_df['bootstrap_iteration'] = i_iter
    return stats_df
