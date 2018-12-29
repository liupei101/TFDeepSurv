# coding=utf-8
"""templates for tuning hyperparams using Bayesian optimization,
read code and change anywhere if necessary.
"""
import sys
import os
import time
import json
import pandas as pd
import numpy as np
import hyperopt as hpt
from sklearn.model_selection import KFold

from tfdeepsurv import dsl
from tfdeepsurv import utils

global Logval, eval_cnt, time_start
global train_X, train_y, validation_X, validation_y
global hidden_layers

########## Configuration for running hyperparams tuning ##########
# Usage: python HyperParametersTuning.py Layer1 Layer2 Layer3 ...

# Don't change it easily
OPTIMIZER_LIST = ['sgd', 'adam']
ACTIVATION_LIST = ['relu', 'tanh']
DECAY_LIST = [1.0, 0.9999]

# Change it before you running
INPUT_FILE_DIR = "data//V3//raw//"
OUTPUT_FILE_DIR = "data//V3//raw//"
SEED = 40
KFOLD = 4
MAX_EVALS = 50
###################################################################

# Change it before you running
def args_trans(args):
    params = {}
    params["num_rounds"] = args["num_rounds"] * 100 + 1500
    params["learning_rate"] = args["learning_rate"] * 0.01 + 0.01
    params["learning_rate_decay"] = DECAY_LIST[args["learning_rate_decay"]]
    params['activation'] = ACTIVATION_LIST[args["activation"]]
    params['optimizer'] = OPTIMIZER_LIST[args["optimizer"]]
    params['L1_reg'] = args["L1_reg"]
    params['L2_reg'] = args["L2_reg"] * 0.001 + 0.005
    params['dropout'] = args["dropout"] * 0.1 + 0.6
    return params

def estimate_time():
    time_now = time.clock()
    total = (time_now - time_start) / eval_cnt * (MAX_EVALS - eval_cnt)
    th = int(total / 3600)
    tm = int((total - th * 3600) / 60)
    ts = int(total - th * 3600 - tm * 60)
    print('Estimate the remaining time: %dh %dm %ds' % (th, tm, ts))

# K-fold cross validation on TFDeepSurv
def train_dsl(args):
    global Logval, eval_cnt

    m = train_X.shape[1]
    params = args_trans(args)
    ci_list = []
    # 4-KFold
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    for train_index, test_index in kf.split(train_X):
        # Split Data(train : test = 3 : 1)
        X_cross_train, X_cross_test = train_X[train_index], train_X[test_index]
        y_cross_train = {'t' : train_y['t'][train_index], 'e' : train_y['e'][train_index]}
        y_cross_test  = {'t' : train_y['t'][test_index],  'e' : train_y['e'][test_index]}
        # Train Network
        ds = dsl.dsnn(
            X_cross_train, y_cross_train,
            m, hidden_layers, 1,
            learning_rate=params['learning_rate'], 
            learning_rate_decay=params['learning_rate_decay'],
            activation=params['activation'],
            optimizer=params['optimizer'],
            L1_reg=params['L1_reg'], 
            L2_reg=params['L2_reg'], 
            dropout_keep_prob=params['dropout']
        )
        ds.train(num_epoch=NUM_EPOCH)
        # Evaluation Network On Test Set
        ci = ds.score(X_cross_test, y_cross_test)
        ci_list.append(ci)
        # Close Session of tensorflow
        ds.close()
        del ds
    # Mean of CI on cross validation set
    ci_mean = sum(ci_list) / KFOLD
    Logval.append({'params': params, 'ci': ci_mean})
    print("Params :", params)
    print(">>> CI :", ci_mean)
    # print remaining time
    eval_cnt += 1
    estimate_time()
    
    return -ci_mean

# Train and validation on TFDeepSurv
def train_dsl_by_vd(args):
    global Logval, eval_cnt

    m = train_X.shape[1]
    params = args_trans(args)
    print("Params: ", params)
    # Train network
    ds = dsl.dsnn(
        train_X, train_y,
        m, hidden_layers, 1,
        learning_rate=params['learning_rate'], 
        learning_rate_decay=params['learning_rate_decay'],
        activation=params['activation'],
        optimizer=params['optimizer'],
        L1_reg=params['L1_reg'], 
        L2_reg=params['L2_reg'], 
        dropout_keep_prob=params['dropout']
    )
    ds.train(num_epoch=params['num_rounds'])
    # Evaluation Network On Test Set
    ci_train = ds.score(train_X, train_y)
    ci_validation = ds.score(validation_X, validation_y)
    # Close Session of tensorflow
    ds.close()
    del ds
    # Mean of CI on cross validation set
    Logval.append({'params': params, 'ci_train': ci_train, 'ci_validation': ci_validation})
    # print remaining time
    eval_cnt += 1
    estimate_time()
    print(">>> CI on train=%g | CI on validation=%g" % (ci_train, ci_validation))

    return -ci_validation

def wt_file(filename, var):
    with open(filename, 'w') as f:
        json.dump(var, f)

def search_params(max_evals = 100):
    global Logval
    # For Real Data
    space = {
        "num_rounds": hpt.hp.randint('num_rounds', 7), # [1500, 2100] = 100 * ([0, 6]) + 1500
        "learning_rate": hpt.hp.randint('learning_rate', 10), # [0.01, 0.10] = 0.01 * ([0, 9] + 1)
        "learning_rate_decay": hpt.hp.randint("learning_rate_decay", 2),# [0, 1]
        "activation": hpt.hp.randint("activation", 2), # [0, 1]
        "optimizer": hpt.hp.randint("optimizer", 2), # [0, 1]
        "L1_reg": hpt.hp.uniform('L1_reg', 0.0, 0.001), # [0.000, 0.001]
        "L2_reg": hpt.hp.randint('L2_reg', 16),  # [0.005, 0.020] = 0.001 * ([0, 15] + 5)
        "dropout": hpt.hp.randint("dropout", 5)# [0.6, 1.0] = 0.1 * ([0, 4] + 6)
    }
    best = hpt.fmin(train_dsl_by_vd, space, algo = hpt.tpe.suggest, max_evals = max_evals)
    wt_file(OUTPUT_FILE_DIR + sys.argv[2], Logval)

    print("best params:", args_trans(best))
    print("best metrics:", -train_dsl_by_vd(best))

def main(filename, use_simulated_data=False):
    
    global Logval, eval_cnt, time_start
    global train_X, train_y, validation_X, validation_y
    global hidden_layers

    if use_simulated_data:
        train_X, train_y = utils.load_simulated_data()
    else:
        # load raw data
        train_X, train_y, validation_X, validation_y = utils.load_raw_data(
            filename,
            out_col = ['patient_id'],
            discount = 0.8,
            seed = SEED
        )
    # assign values for global variables
    Logval = []
    hidden_layers = [int(idx) for idx in sys.argv[3:]]
    eval_cnt = 0
    time_start = time.clock()

    print("Data set for SearchParams: ", len(train_X))
    print("Hidden Layers of Network: ", hidden_layers)
    # start searching params
    search_params(max_evals = MAX_EVALS)

# sys.argv[1] : idfs_train_raw.csv
# sys.argv[2] : hyperopt_log_idfs_train.json
# sys.argv[3:] : 64 32 8
if __name__ == "__main__":
    main(INPUT_FILE_DIR + sys.argv[1], use_simulated_data=False)