from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union
from arcos4py.arcos4py import ARCOS
from arcos4py.tools import calcCollevStats
from tqdm import tqdm


def _get_xy_change(df: pd.DataFrame, object_id_name: str = 'track_id', posCols: list = ['x', 'y']) -> pd.DataFrame:
    """calculate xy change for each object"""
    # get xy change for each object
    df_new = df.copy(deep=True)
    change_cols = [f'{i}_change' for i in posCols]
    df_new[change_cols] = df_new.groupby(object_id_name)[posCols].diff()
    return df_new

def _shuffle_xy_first_frame(df: pd.DataFrame, posCols: list = ['x', 'y'], frame_col: str = 't', seed=42) -> pd.DataFrame:
    """shuffle xy position for each object in the first frame"""
    # get first frame for each object
    first_frame = df[df[frame_col] == df[frame_col].min()].copy(deep=True)
    # shuffle xy position for each object in the first frame
    rng = np.random.default_rng(seed=seed)
    indeces = rng.permutation(len(first_frame))
    first_frame[posCols] = first_frame[posCols].iloc[indeces].to_numpy()
    # merge shuffled first frame with the rest of the data
    df_new = pd.concat([first_frame, df[df[frame_col] > df[frame_col].min()]], ignore_index=True)
    return df_new

def _shuffle_xy_first_frame_track_id(df: pd.DataFrame, posCols: list = ['x', 'y'], frame_col: str = 't', object_id_name='track_id', seed=42) -> pd.DataFrame:
    """shuffle xy position for each object in the first frame"""
    # get first frame for each object
    df_new = df.copy(deep=True)
    new_oid = f'{object_id_name}_temp4shuffle'
    df_new[new_oid] = df_new[object_id_name]
    shuffle_cols = posCols + [new_oid] + [f'{i}_change' for i in posCols]
    first_frame = df_new[df_new[frame_col] == df_new[frame_col].min()].copy(deep=True)
    # shuffle xy position for each object in the first frame
    rng = np.random.default_rng(seed=seed)
    indeces = rng.permutation(len(first_frame))
    first_frame[shuffle_cols] = first_frame[shuffle_cols].iloc[indeces].to_numpy()
    # merge shuffled first frame with the rest of the data
    df_new = pd.concat([first_frame, df_new[df_new[frame_col] > df_new[frame_col].min()]], ignore_index=True)
    return df_new, new_oid


def shuffle_position(df: pd.DataFrame, object_id_name: Union[str, None] = 'track_id', posCols: list = ['x', 'y'], frame_col: str = 't', seed: int = 42) -> pd.DataFrame:
    for i in posCols:
        if i not in df.columns:
            raise ValueError(f'{i} not in df.columns')
    if frame_col not in df.columns:
        raise ValueError('frame_col not in df.columns')
    if object_id_name is None:
        raise ValueError('object_id_name currently needed')
    if object_id_name not in df.columns:
        raise ValueError('object_id_name not in df.columns')
    change_cols = [f'{i}_change' for i in posCols]
    # get xy change for each object
    df_xy_change = _get_xy_change(df, object_id_name, posCols)
    first_frame = df_xy_change[df_xy_change[frame_col] == df_xy_change[frame_col].min()].copy(deep=True)
    df_xy_change.loc[df_xy_change[frame_col] == df_xy_change[frame_col].min(), change_cols] = first_frame[posCols].to_numpy()
    # shuffle xy position for each object in the first frame
    df_shuffled, old_trackid_name = _shuffle_xy_first_frame_track_id(df_xy_change, posCols, frame_col, object_id_name, seed=seed)
    # calculate new xy position for each object
    temp = df_shuffled.groupby(object_id_name)[change_cols].cumsum()
    df_shuffled[posCols] = temp
    df_shuffled.drop(change_cols +[old_trackid_name], axis=1, inplace=True)
    return df_shuffled


def bootstrap_ARCOS_xy(data: pd.DataFrame, posCols: list = ["x"], frame_column: str = 'time', id_column: str = 'id', 
        measurement_column: str = 'meas', clid_column: str = 'clTrackID', smoothK = 3, biasK: int = 51, peakThr: float = 0.2, binThr: float = 0.1, 
        polyDeg: int = 1, biasMet: str = "runmed", eps: float = 2, epsPrev: float | None = None, minClsz: int = 1, nPrev: int = 1, n=1000, seed=42, verbose=False) -> pd.DataFrame:
    for i in posCols:
        if i not in data.columns:
            raise ValueError(f'{i} not in df.columns')
    if frame_column not in data.columns:
        raise ValueError('frame_col not in df.columns')
    if id_column is None:
        raise ValueError('object_id_name currently needed')
    if id_column not in data.columns:
        raise ValueError('object_id_name not in df.columns')
    
    stats_df_list = []
    ts = ARCOS(data, id_column=id_column, posCols=posCols, frame_column=frame_column, measurement_column=measurement_column, clid_column=clid_column)
    ts.interpolate_measurements()
    ts.bin_measurements(smoothK, biasK, peakThr, binThr, polyDeg, biasMet)
    df_arcos = ts.trackCollev(eps, epsPrev, minClsz, nPrev)
    if df_arcos.empty:
        raise ValueError('no events detected in control condition')
    stats_df_original = calcCollevStats().calculate(df_arcos, frame_column, clid_column, id_column, posCols)
    rng = np.random.default_rng(seed=seed)
    integer_seeds = rng.integers(1000000, size=n)
    # shuffle xy position for each object
    print(f'Shuffling position for each object {n} times')
    for i in tqdm(range(n)):
        data_new = shuffle_position(data, object_id_name=id_column, posCols=posCols, frame_col=frame_column, seed=integer_seeds[i])
        if verbose:
            print(f'Calculating ARCOS for shuffled data {i}')
            print(f'Original data: {data}')
            print(f'Shuffled data: {data_new}')
        ts = ARCOS(data_new, id_column=id_column, posCols=posCols, frame_column=frame_column, measurement_column=measurement_column, clid_column=clid_column)
        ts.interpolate_measurements()
        ts.bin_measurements(smoothK, biasK, peakThr, binThr, polyDeg, biasMet)
        df_arcos = ts.trackCollev(eps, epsPrev, minClsz, nPrev)
        if not df_arcos.empty:
            stats_df = calcCollevStats().calculate(df_arcos, frame_column, clid_column, id_column, posCols)
            stats_df['bootstrap_iteration'] = i
            stats_df_list.append(stats_df)
        else:
            if verbose:
                print(f'No events found in shuffled data {i}')

    print('Calculating statistics')
    stats_df = pd.concat(stats_df_list, ignore_index=True)
    print('Calculate significance compared to original data')
    return stats_df, stats_df_original


if __name__ == "__main__":
    data = pd.read_csv('../objNuclei_1line_clean_tracks (1).csv')
    print(data.columns)
    out, yey = bootstrap_ARCOS_xy(data, posCols=['objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y'], 
                                    frame_column='Image_Metadata_T', id_column='track_id', measurement_column='objNucleiS_Intensity_MeanIntensity_imRATIO', 
                                    clid_column='clTrackID', smoothK = 3, biasK = 51, peakThr = 0.5, binThr = 0.01, polyDeg = 1, 
                                    biasMet = "runmed", eps = 35, epsPrev = None, minClsz = 3, nPrev= 1, n=1000, seed=42, verbose=False)

    print(out)
    print(yey)
    print('done')