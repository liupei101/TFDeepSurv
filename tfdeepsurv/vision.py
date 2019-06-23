import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

def plot_train_curve(L, labels=["Learning Curve"], title='Training Curve'):
    """
    Plot model training curve

    Parameters
    ----------
    L: list
        Records list during training.
    labels: list
        Labels of different datasets.
    title: str
        Title of figure.
    """
    if type(L[0]) != list:
        x = range(1, len(L) + 1)
        plt.plot(x, L, label=labels[0])
    else:
        datasets_size = len(L[0])
        for i in range(datasets_size):
            x = range(1, len(L) + 1)
            v = [m_L[i] for m_L in L]
            plt.plot(x, v, label=labels[i])
    # no ticks
    plt.xlabel("Steps")
    plt.legend(loc="best")
    plt.title(title)
    plt.show()

def plot_surv_curve(df_survf, title="Survival Curve"):
    """
    Plot survival curve.

    Parameters
    ----------
    df_survf: DataFrame or numpy.ndarray
        Survival function of samples, shape of which is (n, #Time_Points).
        `Time_Points` indicates the time point presented in columns of DataFrame.
    title: str
        Title of figure.
    """
    if isinstance(df_survf, DataFrame):
        plt.plot(df_survf.columns.values, np.transpose(df_survf.values))
    elif isinstance(df_survf, np.ndarray):
        plt.plot(np.array([i for i in range(df_survf.shape[1])]), np.transpose(df_survf))
    else:
        raise TypeError("Type of arguement is not supported.")

    plt.title(title)
    plt.show()

def plot_km_survf(data, t_col="t", e_col="e"):
    """
    Plot KM survival function curves.

    Parameters
    ----------
    data: pandas.DataFrame
        Survival data to plot.
    t_col: str
        Column name in data indicating time.
    e_col: str
        Column name in data indicating events or status.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    fig, ax = plt.subplots(figsize=(6, 4))
    kmfh = KaplanMeierFitter()
    kmfh.fit(data[t_col], event_observed=data[e_col], label="KM Survival Curve")
    kmfh.survival_function_.plot(ax=ax)
    plt.ylim(0, 1.01)
    plt.xlabel("Time")
    plt.ylabel("Probalities")
    plt.legend(loc="best")
    add_at_risk_counts(kmfh, ax=ax)
    plt.show()
