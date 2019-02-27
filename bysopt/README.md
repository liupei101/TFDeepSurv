## Bayesian Hyperparameters Optimization for TFDeepSurv

This work is based on package [hyperopt](https://github.com/hyperopt/hyperopt), we suggest you know it before you run bayesian hyperparameters optimization for TFDeepSurv.

### When is it needed?

Firstly, when is bayesian hyperparameters optimization needed?

> Whether it runs with simulated data or real-world data, before we intialize our deep survival neural network (i.e., set parameters of `dsl.dsnn`), it is better to do hyperparameter tuning, and then feed the searched optimal hyperparameter to `dsl.dsnn`. It will drastically enhance the performance of deep survival neural network model.

### How does it works?

How does bayesian hyperparameters optimization works?

Here, we give a short introduction. See more details in [hyperopt](https://github.com/hyperopt/hyperopt)

> Given that we have collected the model performance measurements (such as C-index or AUC) under the different setting of hyperparameters' value. The `hyperopt` will generate a new set of hyperparameters' value (in hyperparameter searching space) that may boost the model performance by using the collected historical information.

---

### Step by Step

Open and edit the template file `hpopt.py`. Follow the below steps to prepare the essential part.

What we need to set is located in the place of the snippet `Configuration for Hyperparams Tuning`.

#### 1. Traning Dataset

Firstly, we must prepare dataset for hyperparams tuning. It is a common choice to select training dataset, and preprocess the dataset, such as one-hot encoding and standardization. In the script `hpopt.py`, dataset you provide is splitted into training (for training DNN) and validation set (for measuring model performance).

```python
### Traning Dataset ###
# The folder where the file is located
INPUT_FILE_DIR = "data//"
# The filename
INPUT_FILE_NAME = "data_train.csv"
# The split ratio (80% as training set)
SPLIT_RATIO = 0.8
# The random set for splitting 
SPLIT_SEED = 42
```

#### 2. Network Structure

The network structure (of hidden layers) must be fixed. Or you can take it as hyperparameter and search it.

```python
### Network Structure ###
# Structure of hidden layers 
# Number of neurons in 3 hidden layers are 64, 16, 8, respectively
HIDDEN_LAYERS = [64, 16, 8]
```

#### 3. Search Params

As described in `How does it works?`, the search times and search space must be given. In `hyopt.py`, the search space is set in the simple form of int or float value (as shown in `SEARCH_SPACE`), and then transformed to the value for directly initializing DNN.

```python
### Search Times ###
# The number of iterations for searching
MAX_EVALS = 50

### Search Space ###
# Optimize algorithm for DNN
OPTIMIZER_LIST = ['sgd', 'adam']
# Activation function for DNN
ACTIVATION_LIST = ['relu', 'tanh']
# Learning Rate Decay for DNN
DECAY_LIST = [1.0, 0.9999]
# Simple form of search space (raw form)
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
# Params transformation function (understandable form)
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
```

#### 4. Output Result

The procedure of searching (i.e. the hyperparameters and corresponding model performance in each searching round) is recorded and outputted as a log file.

```python
# The folder where the log file is located
OUTPUT_FILE_DIR = "res//"
# The log filename
OUTPUT_FILE_NAME = "log_hpopt.json"
```

#### 5. Runing the script

Enter the command in your bash or CMD:

```bash
python hpopt.py
```