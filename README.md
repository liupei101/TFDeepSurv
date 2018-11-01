# TFDeepSurv
COX Proportional risk model and survival analysis implemented by tensorflow.

## Differences from DeepSurv
[DeepSurv](https://github.com/jaredleekatzman/DeepSurv) is a package for Deep COX Proportional risk model, published on Github. But our works may differ in:

- Evaluate importance of features in neural network.
- Automatic processing ties of death time in you data, which means different loss function and estimator for survival function.
- Estimator for survival function, three estimating algorithm provided.
- Scientific Hyperparameters tuning method, Bayesian Hyperparameters Optimization for neural network.

## Statement
The paper about this project will be published in this year. The project is based on
research of Breast Cancer, we will update status here once paper published !

## Installation
### From source

Download TFDeepSurv package and install from the directory (**Python version : 3.x**):
```bash
git clone https://github.com/liupei101/TFDeepSurv.git
cd TFDeepSurv
pip install .
```

## Usage:

#### import packages and prepare data:
```python
# import package
from tfdeepsurv import dsl
from tfdeepsurv.dataset import SimulatedData
# generate simulated data
# train data : 2000 rows, 10 features, 2 related features
data_config = SimulatedData(2000, num_var = 2, num_features = 10)
train_data = data_config.generate_data(2000)
# test data : 800 rows
test_data = data_config.generate_data(800)
```

#### Initialize a neural network: you can set some hyperparameters
```python
input_nodes = 10
output_nodes = 1
train_X = train_data['x']
train_y = {'e': train_data['e'], 't': train_data['t']}
model = dsl.dsnn(
    train_X, train_y,
    input_nodes, [6, 3], output_nodes, 
    learning_rate=0.2,
    learning_rate_decay=1.0,
    activation='relu', 
    L1_reg=0.0002, 
    L2_reg=0.0003, 
    optimizer='adam',
    dropout_keep_prob=1.0
)
# Watch if ties occur
# 'noties', 'breslow' when ties occur or 'efron' when ties occur frequently
print(model.get_ties())
```

#### train network:
```python
# Plot curve of loss and CI on train data
model.train(num_epoch=2500, iteration=100,
            plot_train_loss=True, plot_train_ci=True)
```

result :
```
-------------------------------------------------
training steps 1:
loss = 7.08086.
CI = 0.532591.
-------------------------------------------------
training steps 101:
loss = 7.0803.
CI = 0.557864.
-------------------------------------------------
training steps 201:
loss = 7.07884.
CI = 0.591186.
...
...
...
-------------------------------------------------
training steps 2201:
loss = 6.29935.
CI = 0.81826.
-------------------------------------------------
training steps 2301:
loss = 6.30067.
CI = 0.818013.
-------------------------------------------------
training steps 2401:
loss = 6.29985.
CI = 0.818038.
```
Curve of loss and CI:

Loss Value                       | CI
:-------------------------------:|:--------------------------------------:
![](tools/README-loss.png)|![](tools/README-ci.png)

#### evaluate model on data of train and test :
```python
test_X = test_data['x']
test_y = {'e': test_data['e'], 't': test_data['t']}
print("CI on train set: %g" % model.score(train_X, train_y))
print("CI on test set: %g" % model.score(test_X, test_y))
```
result :
```
CI on train set: 0.819224
CI on test set: 0.817987
```

#### evaluate importance of features by weights of neural network
```python
model.get_vip_byweights()
```
result:
```
0th feature score : -0.157754.
1th feature score : 1.
2th feature score : -0.0505626.
3th feature score : -0.0559399.
4th feature score : 0.0426953.
5th feature score : 0.0687309.
6th feature score : 0.00604751.
7th feature score : 0.0584479.
8th feature score : -0.100448.
9th feature score : 0.00362639.
```

#### estimate survival function of patients and plot it
```python
# algo: 'wwe', 'bls' or 'kp', the algorithm for estimating survival function
model.survival_function(test_X[0:3], algo="wwe")
```

result:

![Survival rate](tools/README-surv.png)

## More properties
Scientific Hyperparameters tuning method, Bayesian Hyperparameters Optimization for neural network, which is convenient and automated for tuning hyperparameters in neural network.

For more usage of Bayesian Hyperparameters Optimization, you can see [here](bysopt/README.md)