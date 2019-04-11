from __future__ import print_function
import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
from supersmoother import SuperSmoother

from tfdeepsurv import utils, vision

class dsnn(object):
    def __init__(self, X, label, input_node, hidden_layers_node, output_node,
        learning_rate=0.001, learning_rate_decay=1.0, activation='tanh', 
        L2_reg=0.0, L1_reg=0.0, optimizer='sgd', dropout_keep_prob=1.0, seed=1):
        """dsnn(Deep Survival Neural Network) Class Constructor

        Parameters
        ----------
        X : np.array
            Input data with covariate variables.
        label : dict
            Status and Time of patients in survival analyze,
            example like as {'e': event, 't': time}.
        input_node : int
            Number of input layer.
        hidden_layers_node : list
            Number of nodes in hidden layers of neural network.
        output_node : int
            Number of output layer.
        learning_rate : float
            Learning rate.
        learning_rate_decay : float
            Decay of learning rate.
        activation : string
            Type of activation function. The options include `relu`, `sigmoid` and `tanh`.
        L1_reg : float
            Parameter of L1 regularizate item.
        L2_reg : float
            Parameter of L2 regularizate item.
        optimizer : string
            Type of optimize algorithm. The option include `sgd` and `adam`.
        dropout_keep_prob : float
            Probability of dropout.
        seed : int
            Random state settting.

        Returns
        -------
        `dsnn class`
            An instance of `dsnn class`.

        Examples
        --------
        >>> train_data = load_data()
        >>> train_X = train_data['x']
        >>> train_y = {'e': train_data['e'], 't': train_data['t']}
        >>> model = dsnn(X, y, 117, [64, 32, 8], 1)
        """
        # Prepare data
        self.train_data = dict()
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
                    weights = tf.get_variable(
                        'weights', 
                        [prev_node, hidden_layers_node[i]], 
                        initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
                    self.nnweights.append(weights)

                    biases = tf.get_variable(
                        'biases', 
                        [hidden_layers_node[i]],
                        initializer=tf.constant_initializer(0.0)
                    )

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
                weights = tf.get_variable(
                    'weights', 
                    [prev_node, output_node], 
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                )

                self.nnweights.append(weights)

                biases = tf.get_variable(
                    'biases', 
                    [output_node],
                    initializer=tf.constant_initializer(0.0)
                )

                self.nnbias.append(biases)

                layer_out = tf.matmul(prev_x, weights) + biases
            # Output of Network
            y = layer_out
            # Global step
            with tf.variable_scope('training_step', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable(
                    "global_step", 
                    [], 
                    dtype=tf.int32,
                    initializer=tf.constant_initializer(0), 
                    trainable=False
                )
            # Loss value
            reg_item = tf.contrib.layers.l1_l2_regularizer(L1_reg, L2_reg)
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
                train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)     
            else:
                raise NotImplementedError('activation not recognized')   
            # init op
            init_op = tf.global_variables_initializer()
        
        # Save as class members
        self.X = X
        self.y_ = y_
        self.keep_prob = keep_prob
        self.y = y
        self.global_step = global_step
        self.init_op = init_op
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
        # Create new Session for the DeepSurv Class
        self.sess = tf.Session(graph=G)
        # Initialize all global variables
        self.sess.run(init_op)

    def train(self, num_epoch=5000, iteration=-1, plot_train_loss=False, plot_train_ci=False):
        """Training dsnn.

        Parameters
        ----------
        num_epoch : int
            Number of epoch.
        iteration : int
            After each `iteration`, providing information of training processes.
            
            Default -1, which means keep silence.
        plot_train_loss : bool
            Whether plot curve of loss value during training.
        plot_train_ci : bool
            Whether plot curve of CI on train set during training.

        Returns
        -------
        None

        Examples
        --------
        >>> model.train(num_epoch=2500, iteration=100, plot_train_loss=True, plot_train_ci=True)
        """
        # Record training steps
        loss_list = []
        CI_list = []
        N = self.train_data['E'].shape[0]
        # Train steps
        for i in range(num_epoch):
            _, output_y, loss_value, step = self.sess.run(
                                               [self.train_step, self.y, self.loss, self.global_step],
                                                feed_dict = {
                                                    self.X:  self.train_data['X'],
                                                    self.y_: self.train_data['E'].reshape((N, 1)),
                                                    self.keep_prob: self.configuration['dropout']
                                                }
                                            )
            # Record information
            loss_list.append(loss_value)
            label = {
                't': self.train_data['T'],
                'e': self.train_data['E']
            }
            CI = self._metrics_ci(label, output_y)
            CI_list.append(CI)
            # Print evaluation on test set
            if (iteration != -1) and (i % iteration == 0):
                print("-------------------------------------------------")
                print("On training steps %d:\nloss = %g." % (step, loss_value))
                print("CI = %g." % CI)
        # Plot curve
        if plot_train_loss:
            vision.plot_train_curve(loss_list, title="Loss(train)")

        if plot_train_ci:
            vision.plot_train_curve(CI_list, title="CI(train)")

    def learn(self, num_epoch=5000, iteration=-1, eval_list={}, plot_loss=False, plot_ci=False):
        """Training dsnn and watch the learning curve.

        Parameters
        ----------
        num_epoch : int
            Number of epoch.
        iteration : int
            After each `iteration`, providing information of training processes.
            
            Default -1, means keep silence.
        eval_list: dict
            Survival datasets you want to evaluate or watch during training.

            Default {}. You must set it like {"Data1": [d1_x, d1_y], "Data2": [d2_x, d2_y]}.
            If you set it by default, then the `eval_list` will only include the training data
            which you pass at begining.
        plot_loss : bool
            Whether plot curve of loss value on `eval_list` during training.

            NOTE: This property has not been supported in latest version.
        plot_ci : bool
            Whether plot curve of CI on `eval_list` during training.

        Returns
        -------
        None

        Examples
        --------
        >>> model.learn(
        >>>     num_epoch=2500, iteration=100, 
        >>>     eval_list={"trainset": [train_X, train_y], "testset": [test_X, test_y]},
        >>>     plot_train_loss=True, plot_train_ci=True
        >>> )
        """
        # Clean the environment.
        self.clean()
        # If eval_list is empty, then set it as the traning set.
        if len(eval_list) == 0:
            eval_list = {
                "trainset": [
                    self.train_data['X'], 
                    {
                        't': self.train_data['T'],
                        'e': self.train_data['E']
                    }
                ]
            }
        # Record training steps
        data_name = [k for k in eval_list.keys()]
        # loss_list = {}
        CI_list = {}
        for d in data_name:
            # loss_list[d] = []
            CI_list[d] = []
        N = self.train_data['E'].shape[0]
        # Train steps
        for i in range(num_epoch):
            _, output_y, loss_value, step = self.sess.run(
                                               [self.train_step, self.y, self.loss, self.global_step],
                                                feed_dict = {
                                                    self.X:  self.train_data['X'],
                                                    self.y_: self.train_data['E'].reshape((N, 1)),
                                                    self.keep_prob: self.configuration['dropout']
                                                }
                                            )
            # Record information
            for d in data_name:
                data_X, data_label = eval_list[d]
                # loss_list[d].append(self.loss(data_X, data_label))
                CI_list[d].append(self.score(data_X, data_label))
            # Print result of evaluation on eval_list
            if (iteration != -1) and (i % iteration == 0):
                print("-------------------------------------------------")
                print("On training steps %d:" % step)
                print("\tloss on trainset = %g.\n" % loss_value)
                for d in data_name:
                    print("\tCI on %s: %g." % (d, CI_list[d][-1]))
                
        # Plot curve
        #if plot_loss:
        #    vision.plot_train_curve(loss_list, title="Loss(train)")

        if plot_ci:
            vision.plot_train_curve(CI_list, title="Learning Curve")

    def _negative_log_likelihood(self, y_true, y_pred):
        """Callable loss function for DeepSurv network.

        the negative average log-likelihood of the prediction 
        of this model under a given target distribution.

        Parameters
        ----------
        y_true : tf.tensor
            Observations. 
        y_pred : tf.tensor
            Output of network.

        Returns
        -------
        tf.tensor
            Value of loss function, which means negative log-likelihood.
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
    
    def _metrics_ci(self, label_true, y_pred):
        """Compute the concordance-index value.

        Parameters
        ----------
        label_true : dict
            Status and Time of patients in survival analyze,
            example like as {'e': event, 't': time}.
        y_pred : np.array
            Proportional risk.

        Returns
        -------
        float
            Concordance index.
        """
        hr_pred = -y_pred
        ci = concordance_index(label_true['t'], hr_pred, label_true['e'])
        return ci

    def predict(self, X):
        """Predict risk of X using trained dsnn.

        Parameters
        ----------
        X : np.array
            Input data with covariate variables, shape of which is (n, input_node).

        Returns
        -------
        np.array
            Proportional risk of X, shape of which is (n, ). 

        Examples
        --------
        >>> # "array([0.3, 1.88, -0.1, ..., 0.98])"
        >>> model.predict(test_X)
        """
        # Set dropout to 1.0 when runnig prediction of model
        risk = self.sess.run(
                   [self.y], 
                   feed_dict = {self.X: X, self.keep_prob: 1.0}
               )
        risk = np.squeeze(risk)
        if risk.shape == ():
            risk = risk.reshape((1, ))
        return risk

    def score(self, X, label):
        """Evaluate test set using CI metrics.

        Parameters
        ----------
        X : np.array
            Input data with covariate variables.
        label : dict
            Status and Time of patients in survival analyze,
            example like as {'e': event, 't': time}.

        Returns
        -------
        float
            Score of evaluation on testset.

            CI would be metrics.

        Examples
        --------
        >>> model.score(X, label)
        """
        pred_risk = self.predict(X)
        CI = self._metrics_ci(label, pred_risk)
        return CI

    def close(self):
        """close session of tensorflow.

        Parameters
        ----------

        Returns
        -------
        None

        Examples
        --------
        >>> model.close()
        """
        self.sess.close()
        print("Current session closed!")
    
    def clean(self):
        """Clean all global variables in current graph. But the training data 
        and hyper-parameter setttings will remain the same.

        Parameters
        ----------

        Returns
        -------
        None

        Examples
        --------
        >>> model.clean()
        """
        self.sess.run(self.init_op)
        print("Clean the running state of graph!")

    def get_ties(self):
        """Get the type of ties in train data.
        
        Parameters
        ----------

        Returns
        -------
        string
            Type of ties in train data, includes "noties", "efron" and "breslow".

        Examples
        --------
        >>> model.get_ties()
        """
        return self.train_data['ties']

    def get_vip_byweights(self):
        """Evaluate feature importance by weights of dsnn according to breslow's paper.

        Parameters
        ----------

        Returns
        -------
        np.array
            Value of importance of features.

        Examples
        --------
        >>> model.get_vip_byweights()
        """
        # Fetch weights of network
        W = [self.sess.run(w) for w in self.nnweights]
        n_w = len(W)
        # Matrix multiplication for all hidden layers except last output layer
        hiddenMM = W[- 2].T
        for i in range(n_w - 3, -1, -1):
            hiddenMM = np.dot(hiddenMM, W[i].T)
        # Multiply last layer matrix and compute the sum of each variable for VIP
        last_layer = W[-1]
        s = np.dot(np.diag(last_layer[:, 0]), hiddenMM)

        sumr = s / s.sum(axis=1).reshape(s.shape[0] ,1)
        score = sumr.sum(axis=0)
        VIP = score / score.max()
        for i, v in enumerate(VIP):
            print("%dth feature score : %g." % (i, v))
        return VIP

    def survival_function(self, X, algo="wwe", base_X=None, base_label=None, 
                          smoothed=False, is_plot=True):
        """Estimator of survival function for data X.

        Parameters
        ----------
        X : np.array
            Input data with covariate variables.
        algo : string
            Algorithm for estimating survival function. 

            The options includes "wwe", "kp" and "bsl".
        base_X : np.array
            Input data of patients for estimating survival function.
        base_label : dict
            Input label of patients for estimating survival function.
        smoothed : bool
            Does smooth survival function.

        Returns
        -------
        tuple
            tuple is (T0, ST), T0 of it means time points of survival function, 
            ST of it means survival rate of survival function.

        Examples
        --------
        >>> model.survival_function(X)
        """
        risk = self.predict(X)
        hazard_ratio = np.exp(risk.reshape((risk.shape[0], 1)))
        # Estimate S0(t) using data(base_X, base_label)
        T0, S0 = self.base_surv(algo=algo, X=base_X, label=base_label, smoothed=smoothed)
        ST = S0**(hazard_ratio)
        if is_plot:
            vision.plot_surv_func(T0, ST)
        return T0, ST

    def base_surv(self, algo="bsl", X=None, label=None, smoothed=False):
        """Estimate base survival function S0(t) based on data(X, label).

        Parameters
        ----------
        algo : string
            algorithm for estimating survival function.

            The options includes "wwe", "kp" and "bsl".
        X : np.array
            Input data of patients for estimating survival function.
        label : dict 
            Input label of patients for estimating survival function.
        smoothed : bool
            Does smooth survival function.

        Returns
        -------
        tuple
            tuple is (T0, ST), T0 of it means time points of base survival function, 
            ST of it means survival rate of base survival function.

        Examples
        --------
        >>> model.base_surv(algo='wwe')

        Notes
        -----
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
                    s0.append(np.exp(cj - 1))
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
                    s0.append(np.exp(cj - 1))
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
                    s0.append(np.exp(cj - 1))
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
