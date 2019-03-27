import numpy as np
import matplotlib.pyplot as plt

def plot_train_curve(L, title='Training Curve'):
    if type(L) == list:
        x = range(1, len(L) + 1)
        plt.plot(x, L, label="evaluation set")
    elif type(L) == dict:
        for k, v in L.items():
            x = range(1, len(v) + 1)
            plt.plot(x, v, label=k)
    # no ticks
    plt.xticks([])
    plt.legend(loc="best", title="Datasets")
    plt.title(title)
    plt.show()

def plot_surv_func(T, surRates):
    plt.plot(T, np.transpose(surRates))
    plt.show()