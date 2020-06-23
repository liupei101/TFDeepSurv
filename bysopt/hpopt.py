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

from tfdeepsurv import dsnn
from tfdeepsurv.datasets import load_data, survival_df

global Logval, eval_cnt
global train_X, train_y, validation_X, validation_y

# ignore warning messages from tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#############  Start Configuration for Hyperparams Tuning ###############

### 1. Dataset ###
WORK_DIR = "E:\\My library\\TFDeepSurv\\bysopt"
DATA_PATH = "simulated_data_train.csv"
COLUMN_T = 't'
COLUMN_E = 'e'
IS_NORM = False # data normalization
SPLIT_RATIO = 0.8 # data split for validation
SPLIT_SEED = 42 # random seed

### 2. Model ###
HIDDEN_LAYERS = [7, 3, 1]

### 3. Search ###
MAX_EVALS = 50 # the number of searching or iteration

### 4. Hyperparams Space ###
OPTIMIZER_LIST = ['sgd', 'adam']
ACTIVATION_LIST = ['relu', 'tanh']
DECAY_LIST = [1.0, 0.9999]
SEARCH_SPACE = {
    "num_rounds": hpt.hp.randint('num_rounds', 7), # [1500, 2100] = 100 * ([0, 6]) + 1500
    "learning_rate": hpt.hp.randint('learning_rate', 10), # [0.1, 1.0] = 0.1 * ([0, 9] + 1)
    "learning_rate_decay": hpt.hp.randint("learning_rate_decay", 2),# [0, 1]
    "activation": hpt.hp.randint("activation", 2), # [0, 1]
    "optimizer": hpt.hp.randint("optimizer", 2), # [0, 1]
    "L1_reg": hpt.hp.uniform('L1_reg', 0.0, 0.001), # [0.000, 0.001]
    "L2_reg": hpt.hp.uniform('L2_reg', 0.0, 0.001), # [0.000, 0.001]
    "dropout": hpt.hp.randint("dropout", 3)# [0.8, 1.0] = 0.1 * ([0, 2] + 8)
}
# function for transforming values of hyperparams to a specified range
def args_trans(args):
    params = {}
    params["num_rounds"] = args["num_rounds"] * 100 + 1500
    params["learning_rate"] = args["learning_rate"] * 0.1 + 0.1
    params["learning_rate_decay"] = DECAY_LIST[args["learning_rate_decay"]]
    params['activation'] = ACTIVATION_LIST[args["activation"]]
    params['optimizer'] = OPTIMIZER_LIST[args["optimizer"]]
    params['L1_reg'] = args["L1_reg"]
    params['L2_reg'] = args["L2_reg"]
    params['dropout'] = args["dropout"] * 0.1 + 0.8
    return params

### 5. Output ###
OUTPUT_DIR = "E:\\My library\\TFDeepSurv\\bysopt"
OUTPUT_FILEPATH = "log_hpopt.json"

#############  End Configuration for Hyperparams Tuning ###############


# Training TFDeepSurv model by cross-validation
def train_dsl_by_vd(args):
    global Logval, eval_cnt

    # transform parameters
    m = train_X.shape[1]
    params = args_trans(args)
    
    # train model
    nn_config = {
        "learning_rate": params['learning_rate'], 
        "learning_rate_decay": params['learning_rate_decay'],
        "activation": params['activation'],
        "optimizer": params['optimizer'],
        "L1_reg": params['L1_reg'], 
        "L2_reg": params['L2_reg'], 
        "dropout_keep_prob": params['dropout']
    }
    ds = dsnn(
        m, HIDDEN_LAYERS,
        nn_config
    )
    ds.build_graph()
    ds.train(train_X, train_y, num_steps=params['num_rounds'], silent=True)
    
    # evaluate model on validation set
    ci_train = ds.evals(train_X, train_y)
    ci_validation = ds.evals(validation_X, validation_y)
    
    # close session of tensorflow
    ds.close_session()
    del ds
    
    # append current search record
    Logval.append({'params': params, 'ci_train': ci_train, 'ci_validation': ci_validation})
    
    # print current search params and remaining time
    eval_cnt += 1
    print("[info] After %d-th searching: CI on train=%g | CI on validation=%g" % (eval_cnt, ci_train, ci_validation))

    return -ci_validation

def search_params(max_evals=100):
    # running hyperparams tuning
    space = SEARCH_SPACE
    best = hpt.fmin(train_dsl_by_vd, space, algo=hpt.tpe.suggest, max_evals=max_evals)
    
    # write searching records
    with open(os.path.join(OUTPUT_DIR, OUTPUT_FILEPATH), 'w') as f:
        json.dump(Logval, f)
    
    # print optimal searching result
    print("[result] best params:", args_trans(best))
    print("[result] best metrics:", -train_dsl_by_vd(best))

def main(filepath):
    global Logval, eval_cnt
    global train_X, train_y, validation_X, validation_y

    # load data
    train_data, validation_data = load_data(
        filepath,
        t_col=COLUMN_T,
        e_col=COLUMN_E,
        normalize=IS_NORM,
        split_ratio=SPLIT_RATIO,
        seed=SPLIT_SEED
    )

    # transform dataset to the format of survival data
    train_data = survival_df(train_data, t_col=COLUMN_T, e_col=COLUMN_E, label_col='Y')
    validation_data = survival_df(validation_data, t_col=COLUMN_T, e_col=COLUMN_E, label_col='Y')
    
    # get X and Y (labels)
    columns = list(train_data.columns)
    train_X = train_data[columns[:-1]]
    train_y = train_data[['Y']]
    validation_X = validation_data[columns[:-1]]
    validation_y = validation_data[['Y']]

    # assign values to global variables
    Logval = []
    eval_cnt = 0

    print("Number of Dataset: ", len(train_X))
    print("Hidden Layers of Network: ", HIDDEN_LAYERS)
    
    # start searching params
    search_params(max_evals=MAX_EVALS)

if __name__ == "__main__":
    main(os.path.join(WORK_DIR, DATA_PATH))
    