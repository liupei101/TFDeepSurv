import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
from .dataset import SimulatedData

def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort Training Data for Accurate Likelihood
    # sort array using pandas.DataFrame(According to DESC 't' and ASC 'e')  
    df1 = pd.DataFrame({'t': t, 'e': e})
    df1.sort_values(['t', 'e'], ascending=[False, True], inplace=True)
    sort_idx = list(df1.index)
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, {'e': e, 't': t}

def parse_data(x, label):
    # sort data by t
    x, label = prepare_data(x, label)
    e, t = label['e'], label['t']

    failures = {}
    atrisk = {}
    n, cnt = 0, 0

    for i in range(len(e)):
        if e[i]:
            if t[i] not in failures:
                failures[t[i]] = [i]
                n += 1
            else:
                # ties occured
                cnt += 1
                failures[t[i]].append(i)

            if t[i] not in atrisk:
                atrisk[t[i]] = []
                for j in range(0, i+1):
                    atrisk[t[i]].append(j)
            else:
                atrisk[t[i]].append(i)
    # when ties occured frequently
    if cnt >= n / 2:
        ties = 'efron'
    elif cnt > 0:
        ties = 'breslow'
    else:
        ties = 'noties'

    return x, e, t, failures, atrisk, ties

def load_simulated_data(hr_ratio=2000, n=2000, m=10, num_var=2, seed=1):
    data_config = SimulatedData(hr_ratio, num_var = num_var, num_features = m)
    data = data_config.generate_data(n, seed=seed)
    data_X = data['x']
    data_y = {'e': data['e'], 't': data['t']}
    return data_X, data_y

def load_data(filename, excluded_col=[], surv_col={'e': 'event', 't': 'time'}, 
              split_ratio=1.0, normalize=True, seed=40):
    """
    load csv file and return standard format data for traning or testing.

    Parameters
    ----------
    filename: str
        file name, which only support for csv file.
    excluded_col: list
        columns will not be include in final returned data.
    surv_col: dict
        dict likes {'e': 'xx', 't': 'xxx'}, which is used to indicate columns of time 
        and event in you survival data.
    split_ratio: float, default 1.0
        set `split_ratio` as 1.0, which means returning full data. Otherwise, splitted data 
        will be returned.
    normalize: bool
        If true, then data will be normalized (x - meam / std).
    seed: int
        random seed for splitting data.
    """
    # Read csv data
    data_all = pd.read_csv(filename)

    target = list(surv_col.values())
    L = target + excluded_col
    x_cols = [x for x in data_all.columns if x not in L]

    X = data_all[x_cols]
    y = data_all[target]
    # Normalized data
    if normalize:
        for col in X.columns:
            X.loc[:, col] = (X.loc[:, col] - X.loc[:, col].mean()) / (X.loc[:, col].max() - X.loc[:, col].min())
    # Split data
    if split_ratio == 1.0:
        train_X, train_y = X, y
    else:
        sss = ShuffleSplit(n_splits = 1, test_size = 1-split_ratio, random_state = seed)
        for train_index, test_index in sss.split(X, y):
            train_X, test_X = X.loc[train_index, :], X.loc[test_index, :]
            train_y, test_y = y.loc[train_index, :], y.loc[test_index, :]
    # print information about train data
    print("Number of rows: ", len(train_X))
    print("X cols: ", len(train_X.columns))
    print("Y cols: ", len(train_y.columns))
    print("X.column name:", train_X.columns)
    print("Y.column name:", train_y.columns)
    # Transform type of data to np.array
    train_X = train_X.values
    train_y = {'e': train_y[surv_col['e']].values,
               't': train_y[surv_col['t']].values}
    if split_ratio == 1.0:
        return train_X, train_y
    else:
        test_X = test_X.values
        test_y = {'e': test_y[surv_col['e']].values,
                  't': test_y[surv_col['t']].values}
        return train_X, train_y, test_X, test_y

def load_raw_data(filename, out_col=[], discount=None, seed=1):
    # Get raw data (no split, has been pre-processed)
    data_all = pd.read_csv(filename)
    L = [col for col in data_all.columns if col not in out_col]
    data_all = data_all[L]
    num_features = len(data_all.columns)
    X = data_all.iloc[:, 0:(num_features-2)]
    y = data_all.iloc[:, (num_features-2):]
    # split data
    if discount is None or discount == 1.0:
        train_X, train_y = X, y
    else:
        sss = ShuffleSplit(n_splits=1, test_size=1-discount, random_state=seed)
        for train_index, test_index in sss.split(X, y):
            train_X, test_X = X.loc[train_index, :], X.loc[test_index, :]
            train_y, test_y = y.loc[train_index, :], y.loc[test_index, :]
    # print information about train data
    print("Shape of train_X: ", len(train_X.index), len(train_X.columns))
    print("Shape of train_y: ", len(train_y.index), len(train_y.columns))
    # Transform type of data to np.array
    train_X = train_X.values
    train_y = {'e': train_y.iloc[:, 0].values,
               't': train_y.iloc[:, 1].values}
    if discount is None or discount == 1.0:
        return train_X, train_y
    else:
        # print information about test data
        print("Shape of test_X: ", len(test_X.index), len(test_X.columns))
        print("Shape of test_y: ", len(test_y.index), len(test_y.columns))
        test_X = test_X.values
        test_y = {'e': test_y.iloc[:, 0].values,
                  't': test_y.iloc[:, 1].values}
        return train_X, train_y, test_X, test_y