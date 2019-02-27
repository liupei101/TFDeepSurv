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

from tfdeepsurv import dsl
from tfdeepsurv import utils

global Logval, eval_cnt, time_start
global train_X, train_y, validation_X, validation_y

#############  Configuration for Hyperparams Tuning ###############
### Traning Dataset ###
INPUT_FILE_DIR = "data//"
INPUT_FILE_NAME = "data_train.csv"
SPLIT_RATIO = 0.8
SPLIT_SEED = 42

### Network Structure ###
HIDDEN_LAYERS = [64, 16, 8]

### Search Times ###
MAX_EVALS = 50

### Search Space ###
OPTIMIZER_LIST = ['sgd', 'adam']
ACTIVATION_LIST = ['relu', 'tanh']
DECAY_LIST = [1.0, 0.9999]
SEARCH_SPACE = {
    "num_rounds": hpt.hp.randint('num_rounds', 7), # [1500, 2100] = 100 * ([0, 6]) + 1500
    "learning_rate": hpt.hp.randint('learning_rate', 10), # [0.01, 0.10] = 0.01 * ([0, 9] + 1)
    "learning_rate_decay": hpt.hp.randint("learning_rate_decay", 2),# [0, 1]
    "activation": hpt.hp.randint("activation", 2), # [0, 1]
    "optimizer": hpt.hp.randint("optimizer", 2), # [0, 1]
    "L1_reg": hpt.hp.uniform('L1_reg', 0.0, 0.001), # [0.000, 0.001]
    "L2_reg": hpt.hp.randint('L2_reg', 16),  # [0.005, 0.020] = 0.001 * ([0, 15] + 5)
    "dropout": hpt.hp.randint("dropout", 5)# [0.6, 1.0] = 0.1 * ([0, 4] + 6)
}
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

### Output Result ###
OUTPUT_FILE_DIR = "res//"
OUTPUT_FILE_NAME = "log_hpopt.json"

###################################################################

def estimate_time():
    time_now = time.clock()
    total = (time_now - time_start) / eval_cnt * (MAX_EVALS - eval_cnt)
    th = int(total / 3600)
    tm = int((total - th * 3600) / 60)
    ts = int(total - th * 3600 - tm * 60)
    print('Estimate the remaining time: %dh %dm %ds' % (th, tm, ts))

# Train and validation on TFDeepSurv
def train_dsl_by_vd(args):
    global Logval, eval_cnt
    # Params transformation
    m = train_X.shape[1]
    params = args_trans(args)
    # Train network
    ds = dsl.dsnn(
        train_X, train_y,
        m, HIDDEN_LAYERS, 1,
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
    # Append current search record
    Logval.append({'params': params, 'ci_train': ci_train, 'ci_validation': ci_validation})
    # Print current search params and remaining time
    eval_cnt += 1
    estimate_time()
    print("Params: ", params)
    print(">>> CI on train=%g | CI on validation=%g" % (ci_train, ci_validation))

    return -ci_validation

def search_params(max_evals=100):
    # Hyopt
    space = SEARCH_SPACE
    best = hpt.fmin(train_dsl_by_vd, space, algo=hpt.tpe.suggest, max_evals=max_evals)
    # Output result
    with open(OUTPUT_FILE_DIR + OUTPUT_FILE_NAME, 'w') as f:
        json.dump(Logval, f)
    # Print Optimal search result
    print("best params:", args_trans(best))
    print("best metrics:", -train_dsl_by_vd(best))

def main(filename):
    global Logval, eval_cnt, time_start
    global train_X, train_y, validation_X, validation_y

    # load raw data
    train_X, train_y, validation_X, validation_y = utils.load_raw_data(
        filename,
        discount = SPLIT_RATIO,
        seed = SPLIT_SEED
    )
    # assign values for global variables
    Logval = []
    eval_cnt = 0
    time_start = time.clock()

    print("No. of Samples for Searching Params: ", len(train_X))
    print("Hidden Layers of Network: ", HIDDEN_LAYERS)
    # start searching params
    search_params(max_evals = MAX_EVALS)

if __name__ == "__main__":
    main(INPUT_FILE_DIR + INPUT_FILE_NAME)