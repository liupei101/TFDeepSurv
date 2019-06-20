import numpy as np
import matplotlib.pyplot as plt

def plot_train_curve(L, title='Training Curve'):
    if type(L) == list:
        x = range(1, len(L) + 1)
        plt.plot(x, L, label="Learning Curve")
    elif type(L) == dict:
        for k, v in L.items():
            x = range(1, len(v) + 1)
            plt.plot(x, v, label=k)
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
    df_survf: DataFrame
        Survival function of samples, shape of which is (n, #Time_Points).
        `Time_Points` indicates the time point presented in columns of DataFrame.
    """
    plt.plot(df_survf.columns.values, np.transpose(df_survf.values))
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
    fig, ax = plt.subplots(figsize=(8, 6))
    kmfh = KaplanMeierFitter()
    kmfh.fit(data[t_col], event_observed=data[e_col], label="KM Survival Curve")
    kmfh.survival_function_.plot(ax=ax)
    plt.ylim(0, 1.01)
    plt.xlabel("Time")
    plt.ylabel("Probalities")
    plt.legend(loc="best")
    add_at_risk_counts(kmfh, ax=ax)
    plt.show()
