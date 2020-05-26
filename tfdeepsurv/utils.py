import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index as ci

def _check_input_dimension(dim_nodes, dim_data):
    """
    Check the value of input_nodes and the dimension of data_X. 

    Parameters
    ----------
    dim_nodes: int
        the value of input_nodes.
    dim_data: int
        the dimension of data_X.
    """
    if dim_nodes != dim_data:
        raise ValueError("The value of input nodes must be equal to the number of features.")

def _check_config(config):
    """
    Check configuration and complete it with default_config.

    Parameters
    ----------
    config: dict
        Some configurations or hyper-parameters of neural network.
    """
    default_config = {
        "learning_rate": 0.001,
        "learning_rate_decay": 1.0,
        "activation": "tanh",
        "L2_reg": 0.0,
        "L1_reg": 0.0,
        "optimizer": "sgd",
        "dropout_keep_prob": 1.0,
        "seed": 42
    }
    for k in default_config.keys():
        if k not in config:
            config[k] = default_config[k]

def _check_surv_data(surv_data_X, surv_data_y):
    """
    Check survival data and raise errors.

    Parameters
    ----------
    surv_data_X: DataFrame
        Covariates of survival data.
    surv_data_y: DataFrame
        Labels of survival data. Negtive values are considered right censored.
    """
    if not isinstance(surv_data_X, pd.DataFrame):
        raise TypeError("The type of X must DataFrame.")
    if not isinstance(surv_data_y, pd.DataFrame) or len(surv_data_y.columns) != 1:
        raise TypeError("The type of y must be DataFrame and contains only one column.")

def _prepare_surv_data(surv_data_X, surv_data_y):
    """
    Prepare the survival data. The surv_data will be sorted by abs(`surv_data_y`) DESC.

    Parameters
    ----------
    surv_data_X: DataFrame
        Covariates of survival data.
    surv_data_y: DataFrame
        Labels of survival data. Negtive values are considered right censored. 

    Returns
    -------
    tuple
        sorted indices in `surv_data` and sorted DataFrame.

    Notes
    -----
    For ensuring the correctness of breslow function computation, survival data
    must be sorted by observed time (DESC).
    """
    _check_surv_data(surv_data_X, surv_data_y)
    # sort by T DESC
    T = - np.abs(np.squeeze(np.array(surv_data_y)))
    sorted_idx = np.argsort(T)
    return sorted_idx, surv_data_X.iloc[sorted_idx, :], surv_data_y.iloc[sorted_idx, :]

def concordance_index(y_true, y_pred):
    """
    Compute the concordance-index value.

    Parameters
    ----------
    y_true : np.array
        Observed time. Negtive values are considered right censored.
    y_pred : np.array
        Predicted value.

    Returns
    -------
    float
        Concordance index.
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    t = np.abs(y_true)
    e = (y_true > 0).astype(np.int32)
    ci_value = ci(t, y_pred, e)
    return ci_value

def _baseline_hazard(label_e, label_t, pred_hr):
    ind_df = pd.DataFrame({"E": label_e, "T": label_t, "P": pred_hr})
    summed_over_durations = ind_df.groupby("T")[["P", "E"]].sum()
    summed_over_durations["P"] = summed_over_durations["P"].loc[::-1].cumsum()
    # where the index of base_haz is sorted time from small to large
    # and the column `base_haz` is baseline hazard rate
    base_haz = pd.DataFrame(
        summed_over_durations["E"] / summed_over_durations["P"], columns=["base_haz"]
    )
    return base_haz

def _baseline_cumulative_hazard(label_e, label_t, pred_hr):
    return _baseline_hazard(label_e, label_t, pred_hr).cumsum()

def _baseline_survival_function(label_e, label_t, pred_hr):
    base_cum_haz = _baseline_cumulative_hazard(label_e, label_t, pred_hr)
    survival_df = np.exp(-base_cum_haz)
    return survival_df

def baseline_survival_function(y, pred_hr):
    """
    Estimate baseline survival function by Breslow Estimation.

    Parameters
    ----------
    y : np.array
        Observed time. Negtive values are considered right censored.
    pred_hr : np.array
        Predicted value, i.e. hazard ratio.

    Returns
    -------
    DataFrame
        Estimated baseline survival function. Index of it is time point. 
        The only one column of it is corresponding survival probability.
    """
    y = np.squeeze(y)
    pred_hr = np.squeeze(pred_hr)
    # unpack label
    t = np.abs(y)
    e = (y > 0).astype(np.int32)
    return _baseline_survival_function(e, t, pred_hr)