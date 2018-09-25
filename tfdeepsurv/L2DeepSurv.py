from __future__ import print_function
import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from supersmoother import SuperSmoother

from tfdeepsurv import vision, utils

class L2DeepSurv(object):
    def __init__(self, X, label,
        input_node, hidden_layers_node, output_node,
        learning_rate=0.001, learning_rate_decay=1.0, 
        activation='tanh', 
        L2_reg=0.0, L1_reg=0.0, optimizer='sgd', 
        dropout_keep_prob=1.0,
        seed=1):
        """
        L2DeepSurv Class Constructor.

        Parameters:
            X: np.array, covariate variables.
            label: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
            input_node: int, number of covariate variables.
            hidden_layers_node: list, hidden layers in network.
            output_node: int, number of output.
            learning_rate: float, learning rate.
            learning_rate_decay: float, decay of learning rate.
            activation: string, type of activation function.
            L1_reg: float, coefficient of L1 regularizate item.
            L2_reg: float, coefficient of L2 regularizate item.
            optimizer: string, type of optimize algorithm.
            dropout_keep_prob: float, probability of dropout.
            seed: set random state.
        Returns:
            L2DeepSurv Class.
        """
        # Prepare data
        self.train_data = {}
        self.train_data['X'], self.train_data['E'], \
            self.train_data['T'], self.train_data['failures'], \
            self.train_data['atrisk'], self.train_data['ties'] = utils.parse_data(X, label)
        # New Graph
        G = tf.Graph()
        with G.as_default():
            # Set random state
            tf.set_random_seed(seed)
            # Data input
            X = tf.placeholder(tf.float32, [None, input_node], name = 'x-Input')
            y_ = tf.placeholder(tf.float32, [None, output_node], name = 'label-Input')
            keep_prob = tf.placeholder(tf.float32)
            # hidden layers
            self.nnweights = [] # collect weights of network
            self.nnbias = [] # collect bias of network
            prev_node = input_node
            prev_x = X
            for i in range(len(hidden_layers_node)):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights', [prev_node, hidden_layers_node[i]], 
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.nnweights.append(weights)

                    biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                             initializer=tf.constant_initializer(0.0))
                    self.nnbias.append(biases)

                    layer_out = tf.nn.dropout(tf.matmul(prev_x, weights) + biases, keep_prob)

                    if activation == 'relu':
                        layer_out = tf.nn.relu(layer_out)
                    elif activation == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif activation == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = hidden_layers_node[i]
                    prev_x = layer_out
            # output layers
            layer_name = 'layer_last'
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('weights', [prev_node, output_node], 
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)

                biases = tf.get_variable('biases', [output_node],
                                         initializer=tf.constant_initializer(0.0))
                self.nnbias.append(biases)

                layer_out = tf.matmul(prev_x, weights) + biases
            # Output of Network
            y = layer_out
            # Global step
            with tf.variable_scope('training_step', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable("global_step", [], 
                                              dtype=tf.int32,
                                              initializer=tf.constant_initializer(0), 
                                              trainable=False)
            # Loss value
            reg_item = tf.contrib.layers.l1_l2_regularizer(L1_reg,
                                                           L2_reg)
            reg_term = tf.contrib.layers.apply_regularization(reg_item, self.nnweights)
            loss_fun = self._negative_log_likelihood(y_, y)
            loss = loss_fun + reg_term
            # SGD Optimizer
            if optimizer == 'sgd':
                lr = tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    1,
                    learning_rate_decay
                )
                train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
            elif optimizer == 'adam':
                train_step = tf.train.GradientDescentOptimizer(learning_rate).\
                                                               minimize(loss, global_step=global_step)     
            else:
                raise NotImplementedError('activation not recognized')   
            # init op
            init_op = tf.global_variables_initializer()
        
        # Save into class members
        self.X = X
        self.y_ = y_
        self.keep_prob = keep_prob
        self.y = y
        self.global_step = global_step
        self.loss = loss
        self.train_step = train_step
        self.configuration = {
            'input_node': input_node,
            'hidden_layers_node': hidden_layers_node,
            'output_node': output_node,
            'learning_rate': learning_rate,
            'learning_rate_decay': learning_rate_decay,
            'activation': activation,
            'L1_reg': L1_reg,
            'L2_reg': L2_reg,
            'optimizer': optimizer,
            'dropout': dropout_keep_prob
        }
        # create new Session for the DeepSurv Class
        self.sess = tf.Session(graph=G)
        # Initialize all global variables
        self.sess.run(init_op)

    def train(self, num_epoch=5000, iteration=-1, 
              plot_train_loss=False, plot_train_CI=False):
        """
        Training DeepSurv network.

        Parameters:
            num_epoch: times of iterating whole train set.
            iteration: print information on train set every iteration train steps.
                       default -1, means keep silence.
            plot_train_loss: plot curve of loss value during training.
            plot_train_CI: plot curve of CI on train set during training.

        Returns:

        """
        # Record training steps
        loss_list = []
        CI_list = []
        N = self.train_data['E'].shape[0]
        # Train steps
        for i in range(num_epoch):
            _, output_y, loss_value, step = self.sess.run([self.train_step, self.y, self.loss, self.global_step],
                                                          feed_dict = {self.X:  self.train_data['X'],
                                                                       self.y_: self.train_data['E'].reshape((N, 1)),
                                                                       self.keep_prob: self.configuration['dropout']})
            # Record information
            loss_list.append(loss_value)
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
            CI = self._Metrics_CI(label, output_y)
            CI_list.append(CI)
            # Print evaluation on test set
            if (iteration != -1) and (i % iteration == 0):
                print("-------------------------------------------------")
                print("training steps %d:\nloss = %g.\n" % (step, loss_value))
                print("CI = %g.\n" % CI)
        # Plot curve
        if plot_train_loss:
            vision.plot_train_curve(loss_list, title="Loss(train)")

        if plot_train_CI:
            vision.plot_train_curve(CI_list, title="CI(train)")

    def ties_type(self):
        """
        return the type of ties in train data.
        """
        return self.train_data['ties']

    def predict(self, X):
        """
        Predict risk of X using trained network.

        Parameters:
            X: np.array, shape(n, input_node), covariate variables.

        Returns:
            np.array, shape(n,), Proportional risk of X.
        """
        # Set dropout to 1.0 when runnig prediction of model
        risk = self.sess.run([self.y], feed_dict = {self.X: X, self.keep_prob: 1.0})
        risk = np.squeeze(risk)
        if risk.shape == ():
            risk = risk.reshape((1, ))
        return risk

    def eval(self, X, label):
        """
        Evaluate test set using CI metrics.

        Parameters:
            X: np.array, covariate variables.
            label: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.

        Returns:
            CI, float
        """
        pred_risk = self.predict(X)
        CI = self._Metrics_CI(label, pred_risk)
        return CI

    def close(self):
        """
        close session of tensorflow.
        """
        self.sess.close()
        print("Current session closed!")
    
    def _negative_log_likelihood(self, y_true, y_pred):
        """
        Callable loss function for DeepSurv network.
        the negative average log-likelihood of the prediction
        of this model under a given target distribution.

        Parameters:
            y_true: tensor, observations. 
            y_pred: tensor, output of network.

        Returns:
            loss value, means negative log-likelihood.
        """
        logL = 0
        # pre-calculate cumsum
        cumsum_y_pred = tf.cumsum(y_pred)
        hazard_ratio = tf.exp(y_pred)
        cumsum_hazard_ratio = tf.cumsum(hazard_ratio)
        if self.train_data['ties'] == 'noties':
            log_risk = tf.log(cumsum_hazard_ratio)
            likelihood = y_pred - log_risk
            # dimension for E: np.array -> [None, 1]
            uncensored_likelihood = likelihood * y_true
            logL = -tf.reduce_sum(uncensored_likelihood)
        else:
            # Loop for death times
            for t in self.train_data['failures']:                                                                       
                tfail = self.train_data['failures'][t]
                trisk = self.train_data['atrisk'][t]
                d = len(tfail)
                dr = len(trisk)

                logL += -cumsum_y_pred[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_y_pred[tfail[0]-1])

                if self.train_data['ties'] == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += tf.log(s) * d
                elif self.train_data['ties'] == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0]-1])
                    for j in range(d):
                        logL += tf.log(s - j * r / d)
                else:
                    raise NotImplementedError('tie breaking method not recognized')
        # negative average log-likelihood
        observations = tf.reduce_sum(y_true)
        return logL / observations
    
    def _Metrics_CI(self, label_true, y_pred):
        """
        Compute the concordance-index value.

        Parameters:
            label_true: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
            y_pred: np.array, predictive proportional risk of network.

        Returns:
            concordance index.
        """
        hr_pred = -y_pred
        ci = concordance_index(label_true['t'],
                               hr_pred,
                               label_true['e'])
        return ci

    def evaluate_var_byWeights(self):
        """
        evaluate feature importance by weights of NN.
        """
        # fetch weights of network
        W = [self.sess.run(w) for w in self.nnweights]
        n_w = len(W)
        # matrix multiplication for all hidden layers except last output layer
        hiddenMM = W[- 2].T
        for i in range(n_w - 3, -1, -1):
            hiddenMM = np.dot(hiddenMM, W[i].T)
        # multiply last layer matrix and compute the sum of each variable for VIP
        last_layer = W[-1]
        s = np.dot(np.diag(last_layer[:, 0]), hiddenMM)

        sumr = s / s.sum(axis=1).reshape(s.shape[0] ,1)
        score = sumr.sum(axis=0)
        VIP = score / score.max()
        for i, v in enumerate(VIP):
            print("%dth feature score : %g." % (i, v))
        return VIP

    def survivalRate(self, X, algo="wwe", base_X=None, base_label=None, smoothed=False):
        """
        Estimator of survival function for data X.

        Parameters:
            X: np.array, covariate variables of patients.
            algo: algorithm for estimating survival function.
            base_X: X of patients for estimating survival function.
            base_label: label of patients for estimating survival function.
            smoothed: smooth survival function or not.

        Returns:
            T0: time points of survival function.
            ST: survival rate of survival function.
        """
        risk = self.predict(X)
        hazard_ratio = np.exp(risk.reshape((risk.shape[0], 1)))
        # Estimate S0(t) using data(base_X, base_label)
        T0, S0 = self.basesurv(algo=algo, X=base_X, label=base_label, smoothed=smoothed)
        ST = S0**(hazard_ratio)

        vision.plt_surLines(T0, ST)

        return T0, ST

    def basesurv(self, algo="wwe", X=None, label=None, smoothed=False):
        """
        Estimate base survival function S0(t) based on data(X, label).

        Parameters:
            algo: algorithm for estimating survival function.
            X: X of patients for estimating survival function.
            label: label of patients for estimating survival function.
            smoothed: smooth survival function or not.

        Returns:
            T0: time points of base survival function.
            ST: survival rate of base survival function.
        See:
            Algorithm for estimating basel survival function:
            (1). wwe: WWE(with ties)
            (2). kp: Kalbfleisch & Prentice Estimator(without ties)
            (3). bsl: breslow(with ties, but exists negative value)
        """
        # Get data for estimating S0(t)
        if X is None or label is None:
            X = self.train_data['X']
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
        X, E, T, failures, atrisk, ties = utils.parse_data(X, label)

        s0 = [1]
        t0 = [0]
        risk = self.predict(X)
        hz_ratio = np.exp(risk)
        if algo == 'wwe':
            for t in T[::-1]:
                if t in t0:
                    continue
                t0.append(t)
                if t in atrisk:
                    # R(t_i) - D_i
                    trisk = [j for j in atrisk[t] if j not in failures[t]]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / (dt + s)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'kp':
            for t in T[::-1]:
                if t in t0:
                    continue
                t0.append(t)
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    s = np.sum(hz_ratio[trisk])
                    si = hz_ratio[failures[t][0]]
                    cj = (1 - si / s) ** (1 / si)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'bsl':
            for t in T[::-1]:
                if t in t0:
                    continue
                t0.append(t)
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / s
                    s0.append(cj)
                else:
                    s0.append(1)
        else:
            raise NotImplementedError('tie breaking method not recognized')
        # base survival function
        S0 = np.cumprod(s0, axis=0)
        T0 = np.array(t0)

        if smoothed:
            # smooth the baseline hazard
            ss = SuperSmoother()
            #Check duplication points
            ss.fit(T0, S0, dy=100)
            S0 = ss.predict(T0)

        return T0, S0

    def div_three_groups(self, data):
        if isinstance(data, dict):
            X, E, T = data['x'], data['e'], data['t']
        # predict risk
        risk = self.predict(X)
        hr_pred = np.exp(risk)
        # Cut-off1
        cutoff = utils.get_cutoff(hr_pred, T, E)
        ct1 = hr_pred >= cutoff

        X1, X2 = hr_pred[ct1], hr_pred[~ct1]
        T1, T2 = T[ct1], T[~ct1]
        E1, E2 = E[ct1], E[~ct1]
        
        # cf1 cut for X1, cf2 cut for X2 
        cutoff1 = utils.get_cutoff(X1, T1, E1)
        cutoff2 = utils.get_cutoff(X2, T2, E2)

        Hgp = (hr_pred >= cutoff1)
        Lgp = (hr_pred < cutoff2)
        Mgp = (~Hgp) & (~Lgp)

        print('Number of high risk group :', np.sum(Hgp))
        print('          middle risk group :', np.sum(Mgp))
        print('          low risk group :', np.sum(Lgp))
        Th = np.asarray(T[Hgp])
        Eh = np.asarray(E[Hgp])
        Tm = np.asarray(T[Mgp])
        Em = np.asarray(E[Mgp])
        Tl = np.asarray(T[Lgp])
        El = np.asarray(E[Lgp])

        vision.plt_riskGroups(Th, Eh, Tl, El, Tm, Em)

        # logrank test
        summary12_ = logrank_test(Th, Tm, Eh, Em, alpha=0.99)
        summary11_ = logrank_test(Tl, Tm, El, Em, alpha=0.99)

        print(summary12_, summary11_)
        print('______')