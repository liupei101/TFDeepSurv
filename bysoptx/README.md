## Bayesian Hyperparameters Optimization for TFDeepSurv

This work is based on package [hyperopt](https://github.com/hyperopt/hyperopt), we suggest you view it before you run bayesian hyperparameters optimization for TFDeepSurv.

### Usage

Run hpopt.py, and set filename of input and log, number of nodes in hidden layers.

```bash
python hpopt.py inputdata.csv log.json 16 8 4 2
```

You must change follows before you run:

```python
# Space for searching params
space = {
  "learning_rate": hpt.hp.randint('learning_rate', 10), # [0.01, 0.10] = 0.01 * ([0, 9] + 1)
  "learning_rate_decay": hpt.hp.randint("learning_rate_decay", 2),# [0, 1]
  "activation": hpt.hp.randint("activation", 2), # [0, 1]
  "optimizer": hpt.hp.randint("optimizer", 2), # [0, 1]
  "L1_reg": hpt.hp.uniform('L1_reg', 0.0, 0.001), # [0.000, 0.001]
  "L2_reg": hpt.hp.randint('L2_reg', 16),  # [0.005, 0.020] = 0.001 * ([0, 15] + 5)
  "dropout": hpt.hp.randint("dropout", 5)# [0.6, 1.0] = 0.1 * ([0, 4] + 6)
}

# Candidates for OPTIMIZER, ACTIVATION and LEARNING_RATE_DECAY
OPTIMIZER_LIST = ['sgd', 'adam']
ACTIVATION_LIST = ['relu', 'tanh']
DECAY_LIST = [1.0, 0.9999]

# Change it before you running
SEED = 40
KFOLD = 4
MAX_EVALS = 20
NUM_EPOCH = 2400

# Target function for evaluating network
trainVdDeepSurv() # evaluate use train and validation set on TFDeepSurv
trainDeepSurv()   # evaluate use K-fold cross validation  on TFDeepSurv

# Train data, you can define you own function to load data
train_X, train_y, validation_X, validation_y = utils.loadRawData(
  filename = "data//train.csv",
  discount = 0.8,
  seed = SEED
)
```