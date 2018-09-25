import numpy as np
import random
import pandas as pd
import pyper as pr
from sklearn.model_selection import ShuffleSplit, train_test_split
from tfdeepsurv import vision
from tfdeepsurv.dataset import SimulatedData

def get_cutoff(X, T, E):
    # survivalROC to estimate the cutoff for risk groups, fpr = 1-specifity, tpr = sensitivity
    r = pr.R(use_pandas=True)
    r.assign("t", T)
    r.assign("e", E)
    r.assign("mkr", np.reshape(X, E.shape))
    r.assign("pt", 104)
    r.assign("mtd", "KM")
    r.assign("nobs", X.shape[0])

    r("library(survivalROC)")
    r("src <- survivalROC(Stime = t, status = e, marker = mkr, predict.time = pt, span = 0.25*nobs^(-0.20))")
    r("Yuden <- src$TP-src$FP")
    r("cutoff <- src$cut.values[which(Yuden == max(Yuden), arr.ind = T)]")
    r("abline(0,1)")
    r("tpv <- src$TP[which(Yuden == max(Yuden), arr.ind = T)]")
    print( 'cutoff = ', r.cutoff, 'tpv = ', r.tpv)
    vision.plt_survROC(r.src['FP'], r.src['TP'])
    return r.cutoff

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

def loadSimulatedData(hr_ratio=2000, n=2000, m=10, num_var=2, seed=1):
    data_config = SimulatedData(hr_ratio, num_var = num_var, num_features = m)
    data = data_config.generate_data(n, seed=seed)
    data_X = data['x']
    data_y = {'e': data['e'], 't': data['t']}
    return data_X, data_y

def loadData(filename = "data//surv_aly_idfs.csv", 
             tgt={'e': 'idfs_bin', 't': 'idfs_month'}, 
             split=1.0,
             Normalize=True,
             seed=40):
    data_all = pd.read_csv(filename)

    ID = 'patient_id'
    target = list(tgt.values())
    L = target + [ID]
    x_cols = [x for x in data_all.columns if x not in L]

    X = data_all[x_cols]
    y = data_all[target]
    # Normalized data
    if Normalize:
        for col in X.columns:
            X.loc[:, col] = (X.loc[:, col] - X.loc[:, col].mean()) / (X.loc[:, col].max() - X.loc[:, col].min())
    # Split data
    if split == 1.0:
        train_X, train_y = X, y
    else:
        sss = ShuffleSplit(n_splits = 1, test_size = 1-split, random_state = seed)
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
    train_y = {'e': train_y[tgt['e']].values,
               't': train_y[tgt['t']].values}
    if split == 1.0:
        return train_X, train_y
    else:
        test_X = test_X.values
        test_y = {'e': test_y[tgt['e']].values,
                  't': test_y[tgt['t']].values}
        return train_X, train_y, test_X, test_y

def loadRawData(filename, 
                out_col=[],
                discount=None,
                seed=1):
    # Get raw data(no split, has been pre-processed)
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