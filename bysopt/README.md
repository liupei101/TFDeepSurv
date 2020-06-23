## Bayesian Hyperparameters Optimization for TFDeepSurv

Since this work is based on [hyperopt](https://github.com/hyperopt/hyperopt), we suggest you know it before you run bayesian hyperparameters optimization for TFDeepSurv.

### When is it needed?

Firstly, when is bayesian hyperparameters optimization needed?

> Whether it runs with simulated data or real-world data, before we intialize our deep survival neural network (i.e., set parameters of `dsl.dsnn`), it is better to do hyperparameter tuning, and then feed the searched optimal hyperparameter to `dsl.dsnn`. It will drastically enhance the performance of deep survival neural network model.

### How does it work?

How does bayesian hyperparameters optimization work?

Here, we give a short introduction. See more details in [hyperopt](https://github.com/hyperopt/hyperopt)

> Given that we have collected the model performance measurements (such as C-Index or AUC) under the different setting of hyperparameters' value. The `hyperopt` will generate a new set of hyperparameters' value (in hyperparameter searching space) that may boost the model performance by using the collected historical information.

---

### Step by Step

Open and edit the template file `hpopt.py`. Follow the below steps to prepare the essential part.

What we need to set is located in the place of the snippet `Start Configuration for Hyperparams Tuning`.

#### 1. Dataset

Firstly, we must prepare dataset for hyperparams tuning. It is a common choice to select training dataset, and preprocess the dataset, such as one-hot encoding and normalization. In the script `hpopt.py`, dataset you provide is splitted into training (for training model) and validation set (for measuring model performance).

```python
### 1. Dataset ###
WORK_DIR = "E:\\My library\\TFDeepSurv\\bysopt"
DATA_PATH = "simulated_data_train.csv"
COLUMN_T = 't'
COLUMN_E = 'e'
IS_NORM = False # data normalization
SPLIT_RATIO = 0.8 # data split for validation
SPLIT_SEED = 42 # random seed
```

#### 2. Model

The network structure (of hidden layers) must be fixed. Otherwise, you can take it as a hyperparameter and search it, but you need to implement it by yourself.

```python
### 2. Model ###
HIDDEN_LAYERS = [6, 3, 1]
```

#### 3. Search

As described in `How does it works?`, the searching times and searching space must be given. In `hyopt.py`, the searching space is set to a simple combination of int or float value (as shown in `SEARCH_SPACE`), and then transformed to the expected range.

```python
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
```

#### 4. Output

Here we record the searching result and write it to local files.

After appling the [simulated training data](https://github.com/liupei101/TFDeepSurv/tree/master/bysopt/simulated_data_train.csv) to test the script `hpopt.py`, the output file [log_hpopt.json](https://github.com/liupei101/TFDeepSurv/tree/master/bysopt/log_hpopt.json) was generated.

```python
### 5. Output ###
OUTPUT_DIR = "E:\\My library\\TFDeepSurv\\bysopt"
OUTPUT_FILEPATH = "log_hpopt.json"
```

#### 5. Runing the script

Enter the command in your terminal (for Linux) or CMD (for Windows):

```bash
python3 hpopt.py
```
