"""Tools for resampling data and bootstrapping."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from arcos4py import ARCOS
from arcos4py.tools import calcCollevStats

from ._resampling import resample_data


def bootstrap_arcos(
    df: pd.DataFrame,
    posCols: list,
    frame_column: str,
    id_column: str,
    meas_column: str,
    method: str | list[str] = 'shuffle_tracks',
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
    stats_metric: str | list[str] = ["total_size", "duration"],
    n: int = 100,
    seed: int = 42,
    show_progress: bool = True,
    verbose: bool = False,
    paralell_processing: bool = True,
    plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap data using the ARCOS algorithm.

    Arguments:
        df: DataFrame containing the data to be bootstrapped.
        posCols: List of column names containing the x and y coordinates.
        frame_column: Name of the column containing the frame number.
        id_column: Name of the column containing the track id.
        meas_column: Name of the column containing the measurement.
        method: Method used for bootstrapping. Can be "shuffle_tracks", 'shuffle_timepoints', 'shift_timepoints',
            'shuffle_binary_blocks', 'shuffle_coordinates_timepoint or a list of methods,
            which will be applied in order of index.
        smoothK: Smoothing kernel size.
        biasK: Bias kernel size.
        peakThr: Threshold for peak detection.
        binThr: Threshold for binarization.
        polyDeg: Degree of the polynomial used for bias correction.
        biasMet: Bias correction method. Can be 'none', 'runmed', 'lm'
        eps: Epsilon parameter for DBSCAN.
        epsPrev: Parameter for linking tracks. If None, eps is used.
        minClsz: Minimum cluster size.
        nPrev: Number of previous frames to consider for linking.
        stats_metric: Metric to calculate. Can be "duration", "total_size", "min_size", "max_size" or a list of metrics.
            Default is ["duration", "total_size"].
        n: Number of bootstraps.
        seed: Seed for the random number generator.
        show_progress: Show a progress bar.
        verbose: Print additional information.

    Returns:
        DataFrame containing the bootstrapped data.
    """
    if not isinstance(stats_metric, list):
        stats_metric = [stats_metric]

    for stats_m in stats_metric:
        if stats_m not in [
            "duration",
            "total_size",
            "min_size",
            "max_size",
        ]:
            raise ValueError(f"Invalid metric: {stats_metric}")

    clid_name = 'clid'
    stats_df_list = []

    if isinstance(method, str):

        print(f'Resampling data using method "{method}"...')
    elif isinstance(method, list):
        print(f'Resampling data using methods "{method}"...')

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

    print(f'Running ARCOS and calculating "{stats_metric}"...')

    if paralell_processing:
        from joblib import Parallel, delayed

        stats_df_list = Parallel(n_jobs=-1)(
            delayed(_apply_arcos)(
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
            for i_iter in tqdm(iterations, disable=not show_progress)
        )
    else:
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

    stats_df_indexer = ['bootstrap_iteration'] + stats_metric
    stats_df_mean: pd.DataFrame = (
        stats_df[stats_df_indexer].groupby(['bootstrap_iteration']).agg(['mean']).reset_index()
    )
    stats_df_mean = stats_df_mean.droplevel(level=1, axis=1)
    # for bootstrap iteratoins that did not detect any events, set the metric to 0
    stats_df_mean[stats_metric] = stats_df_mean[stats_metric].fillna(0)

    pval = stats_df_mean[stats_metric].agg(_p_val_finite_sampling)
    pval.name = 'p_value'
    df_p = pd.DataFrame(pval)

    if plot:
        fig, axis = plt.subplots(1, 2, sharey=True)
        for ax, stats_col in zip(axis, stats_df_mean.columns[1:]):
            ax.hist(stats_df_mean[stats_col], bins=len(stats_df_mean), alpha=0.5)
            ax.set_title(stats_col)
            ax.vlines(stats_df_mean[stats_col].iloc[0], ymin=0, ymax=ax.get_ylim()[1], color='red', ls='--')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.text(
                ax.get_xlim()[1] * 0.8,
                ax.get_ylim()[1] * 0.9,
                f'p-value: \n {df_p.loc[stats_col, "p_value"]:.3f}',
                ha='center',
                va='center',
                color='red',
            )
        fig.suptitle('Bootstrapped metrics')
        plt.show()
    # p_val = lambda x: (sum(x[1:] > x[0])) / (len(x[1:]))

    # p_value = sum(means > mean_0) / len(means)
    return stats_df, df_p


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
    epsPrev: float | None,
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


def _p_val_finite_sampling(x: pd.DataFrame):
    return (1 + sum(x[1:] > x[0])) / (1 + len(x[1:]))
