import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

def plot_train_curve(L, title='Training Curve'):
	x = range(1, len(L) + 1)
	plt.plot(x, L)
	# no ticks
	plt.xticks([])
	plt.title(title)
	plt.show()

def plt_surLines(T, surRates):
    plt.plot(T, np.transpose(surRates))
    plt.show()

def plt_survROC(fp, tp):
    plt.subplot()
    plt.plot(fp, tp)
    plt.show()

def plt_riskGroups(Th, Eh, Tl, El, Tm=[], Em=[]):
    ax = plt.subplot()

    kmfh = KaplanMeierFitter()
    kmfh.fit(Th, event_observed=Eh, label='High risk group')
    kmfh.plot(ax=ax, ci_force_lines=False)

    kmfl = KaplanMeierFitter()
    kmfl.fit(Tl, event_observed=El, label='Low risk group')
    kmfl.plot(ax=ax, ci_force_lines=False)

    if not (Tm == [] and Em == []):
        kmfm = KaplanMeierFitter()
        kmfm.fit(Tm, event_observed=Em, label='Middle risk group')
        kmfm.plot(ax=ax, ci_force_lines=False)

    plt.ylim(0, 1)
    plt.show()