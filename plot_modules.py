import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.colors as mc
from scipy.optimize import minimize
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def weibull(x, x0, a, lambda_, k):
    y = np.zeros_like(x)
    y = (k/lambda_)*(((x-x0)/lambda_)**(k-1))*np.exp(-((x-x0)/lambda_)**k)
    y[x < x0] = 0
    return a*y

def objfun_weibull(params, x, y):
    x0, a, lambda_, k = params
    f_theo = weibull(x, x0, a, lambda_, k)
    return np.sqrt(np.mean((y-f_theo)**2))

def objfun_gaussian(params, x, y):
    a, x0, sigma = params
    f_theo = gaussian(x, a, x0, sigma)
    return np.sqrt(np.mean((y-f_theo)**2))

def plot_predictions(df, prob, figure_name='prediction.png', dt=30):
    """ function to produce a conprehensive plot of the predictions realized
        by the selected Random Forest model.
    """

    pred = np.round(prob)

    """ fit weibull functions for the mainshock and aftershock distribution """
    # -- mainshock distribution
    hb, bins = np.histogram(np.log10(df[pred == 0]['N+']),
                            np.linspace(-15, 1, 100))
    pm = minimize(fun=objfun_weibull,
                    x0=(-8, 500, 6, 8),
                    method='Nelder-Mead',
                    args=(bins[:-1], hb))
    x = bins[0]+np.cumsum(np.diff(bins))
    fit_main = weibull(x=x, x0=pm.x[0], a=pm.x[1], lambda_=pm.x[2], k=pm.x[3])

    # -- aftershock distribution
    ha, _ = np.histogram(np.log10(df[pred == 1]['N+']), bins=bins)
    pa = minimize(fun=objfun_weibull,
                    x0=(-12, 500, 6, 8),
                    method='Nelder-Mead',
                    args=(bins[:-1], ha))
    fit_after = weibull(x=x, x0=pa.x[0], a=pa.x[1], lambda_=pa.x[2], k=pa.x[3])

    """ figure """
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(6, 2)
    ax1 = fig.add_subplot(gs[:3, 0])
    ax2 = fig.add_subplot(gs[:3, 1])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4:, :])
    pie1 = fig.add_axes([0.37, 0.86, 0.12, 0.12])

    """ this create a colormap which alpha varies with the probability to be
        an aftershock or a mainshock.
    """
    color = '#f27149'
    cmap_main = mc.LinearSegmentedColormap.from_list(
            'incr_alpha', [(0, (*mc.to_rgb(color), 0)), (1, color)])
    color = '#f2e35e'
    cmap_after = mc.LinearSegmentedColormap.from_list(
            'incr_alpha', [(0, (*mc.to_rgb(color), 0)), (1, color)])

    """ plot rescaled distances subplot """
    ax1.scatter(df[pred == 0]['T+'], df[pred == 0]['R+'], s=2,
        c=1-prob[pred == 0], cmap=cmap_main, ec='None')
    ax1.scatter(df[pred == 1]['T+'], df[pred == 1]['R+'], s=2,
        c=prob[pred == 1], cmap=cmap_after, ec='None')

    # -- plot classic approach threshold
    threshold = 3.7*1e-6
    tij = np.array([1e-9, 1e4], dtype=float)
    rij = threshold/tij
    ax1.plot(tij, rij, '--', c='#787878')

    # --  scatter plot for legend (trick)
    ax1.scatter(0, 0, s=100, fc='#f27149', ec='None', label='mainshocks')
    ax1.scatter(0, 0, s=100, fc='#f2e35e', ec='None', label='aftershocks')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-9, 1e4)
    ax1.set_ylim(1e-10, 10)
    ax1.set_ylabel(r'Rescaled Distance $R_{ij}$')
    ax1.set_xlabel(r'Rescaled Time $T_{ij}$')
    ax1.legend(loc='lower left', frameon=False)

    """ plot nearest-neighbor-distance distribution """
    x = [df[pred == 0]['N+'], df[pred == 1]['N+']]
    ax2.hist(x, bins=10**np.linspace(-15, 1, 100), stacked=True,
            color=['#f27149', '#f2e35e'])

    # -- plot fit
    x = bins[0]+np.cumsum(np.diff(bins))
    ax2.semilogx(10**x, fit_main, 'k--')
    ax2.semilogx(10**x, fit_after, 'k--', label='Weibull fit')
    ax2.semilogx(10**x, fit_after+fit_main, 'k-', lw=2)

    # -- plot classic approach threshold
    ax2.axvline(threshold, ls='--', c='#787878')

    ax2.tick_params(left=False, labelleft=False, right=True, labelright=True)
    ax2.set_xlabel(r'Nearest-Neighbor Distance $\eta_{ij}$')
    ax2.set_ylabel('counts')
    ax2.yaxis.set_label_position("right")

    """ plot pdf of the probability to be an aftershock """
    # -- this is ugly but does the job
    h, bins = np.histogram(prob, bins=np.linspace(0, 1, 20))
    h = np.array(h, dtype=float)
    h /= h.sum()
    bins = bins[:-1]
    bins[-1] = 1

    k = bins <= 0.5
    h_background = h[k]
    h_aftershocks = h[~k]

    ax3.plot(bins, h, c='#606060')

    for i in range(len(bins)-1):
        b = bins[i:i+2]
        h_ = h[i:i+2]
        if b[0] <= 0.5:
            color = '#f27149'
            alpha = 1 - b[0] / 0.5
        else:
            color = '#f2e35e'
            alpha = (b[0] - 0.5)/ 0.5
        ax3.fill_between(b, h_, fc=color, ec='None', alpha=alpha)
    ax3.axvline(0.5, linestyle='--', color='k')
    ax3.set_ylim(0, 0.52)
    ax3.set_xlim(0, 1.0)

    font = FontProperties()
    font.set_style('italic')
    font.set_weight('bold')
    font.set_size('13')

    ax3.text(0.2, ax3.get_ylim()[1]/1.75, 'background', horizontalalignment='right',
    verticalalignment='center', fontsize=16, fontproperties=font)
    ax3.text(0.8, ax3.get_ylim()[1]/1.75, 'aftershock', horizontalalignment='left',
    verticalalignment='center', fontsize=16, fontproperties=font)
    font.set_size('11')
    ax3.set_xlabel('probability to be an aftershock')
    ax3.set_ylabel('pdf')

    """ plot dt count of earthquake through time """
    x = [df[pred == 0]['time'], df[pred == 1]['time']]
    bins = np.arange(int(df.time.min()), int(df.time.max())+dt, dt)
    ax4.hist(x, bins=bins, stacked=True,
        edgecolor='None', color=['#f27149', '#f2e35e'])
    ax4.set_yscale('log')
    ax4.set_xlim(int(df.time.min()), int(df.time.max()))
    ax4.set_ylim(1, 5e3)
    ax4.set_ylabel('{} day(s) counts'.format(dt))

    majlocator = md.YearLocator(base=4)
    minlocator = md.MonthLocator(interval=2)
    formatter = md.DateFormatter('%Y')
    ax4.xaxis.set_minor_locator(minlocator)
    ax4.xaxis.set_major_locator(majlocator)
    ax4.xaxis.set_major_formatter(formatter)

    n_background = len(df[pred == 0])
    n_aftershocks = len(df[pred == 1])
    n_total = len(df)
    pie1.pie([n_background/n_total, n_aftershocks/n_total],
        colors=['#f27149', '#f2e35e'], autopct='%.0f%%',
        wedgeprops={"edgecolor":"k",'linewidth': 1.5, 'antialiased': True})

    ax2.legend(loc='upper left', frameon=False)
    plt.savefig(figure_name)
    plt.close()
