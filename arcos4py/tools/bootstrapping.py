from random import shuffle
import pandas as pd
import numpy as np


### calculate per frame change in xy for each object
def get_xy_change(df: pd.DataFrame, object_id_name: str = 'track_id', x_col_name: str = 'x', y_col_name= 'y') -> pd.DataFrame:
    # get xy change for each object
    df_new = df.copy(deep=True)
    df_new[['x_change', 'y_change']] = df_new.groupby(object_id_name)[[x_col_name, y_col_name]].diff()
    return df_new

### shuffle xy position for each object in the first frame
def shuffle_xy(df: pd.DataFrame, x_col_name: str = 'x', y_col_name= 'y', frame_col: str = 't') -> pd.DataFrame:
    # get first frame for each object
    first_frame = df[df[frame_col] == df[frame_col].min()].copy(deep=True)
    # shuffle xy position for each object in the first frame
    rng = np.random.default_rng()
    indeces = rng.permutation(len(first_frame))
    first_frame[[x_col_name, y_col_name]] = first_frame[[x_col_name, y_col_name]].iloc[indeces].to_numpy()
    # merge shuffled first frame with the rest of the data
    df_new = pd.concat([first_frame, df[df[frame_col] > df[frame_col].min()]], ignore_index=True)
    return df_new

### bootstrap xy position for each object
def bootstrap_xy(df: pd.DataFrame, object_id_name: str = 'track_id', x_col_name: str = 'x', y_col_name= 'y', frame_col: str = 't') -> pd.DataFrame:
    # shuffle xy position for each object in the first frame
    df = shuffle_xy(df, x_col_name, y_col_name, frame_col)
    # get xy change for each object
    df = get_xy_change(df, object_id_name, x_col_name, y_col_name)
    # calculate new xy position for each object
    first_frame = df[df[frame_col] == df[frame_col].min()].copy(deep=True)
    df.loc[df[frame_col] == df[frame_col].min(), ['x_change', 'y_change']] = first_frame[[x_col_name, y_col_name]].to_numpy() 
    df[[x_col_name, y_col_name]] = df.groupby(object_id_name)[['x_change', 'y_change']].cumsum()
    df.drop(['x_change', 'y_change'], axis=1, inplace=True)
    return df

data = pd.read_csv('notebooks/sample_data/arcos_data.csv')

data_new = bootstrap_xy(data, object_id_name='id', x_col_name='x', y_col_name='y', frame_col='t')
print('done')