
from typing import Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Some display settings for numpy Array and pandas DataFrame
np.set_printoptions(precision=4, linewidth=94, suppress=True)
pd.set_option('display.max_columns', None)


def fah(
        x: Union[Sequence[float], np.ndarray, pd.Series],
        bins: int = 10,
        graph: bool = True,
        figsize: Tuple[float, float] = (11, 6),
        xlabel: str = ''
    ) -> Union[mpl.figure.Figure, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    **F**requency **A**nalisys for **H**igh values (fah)

    Parameters
    ----------
    x : array-like (1d)
        One dimensional array-like of float type
    bins : int, default=10
        The number of equal-width bins in the range.
    graph : bool, default=True
        Output as a figure (True), a tuple of two DataFrames otherwise.
    figsize : tuple, default=(11, 6), in case `graph=True`
        The size of the output figure.
    xlabel : str, default=''
        The customised string for x label.

    Returns
    -------
    matplotlib.figure.Figure (`graph=True`)
        A plot with fitted distribution with observed frequency.
    A tuple of two DataFrames (`graph=False`):
        1. The DataFrame of distribution parameters with an ascending order of SSEs\n
        2. The estimated values from different annual return periods.

    Notes
    -----
    1. This function is simply used the SSE (sum of squared errors) as a fit of goodness\n
    2. The distributions (all from existing from `scipy.stats`) used:
        * 'Normal': stats.norm,
        * 'Lognorm': stats.lognorm,
        * 'Exponential': stats.expon,
        * 'Gamma': stats.gamma,
        * 'GEV': stats.genextreme,
        * 'Right-skewed Gumbel': stats.gumbel_r,
        * 'Generalized Pareto': stats.genpareto,
        * 'Pearson Type III': stats.pearson3,

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/reference/stats.html
    """
    x = np.array(x)
    sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
    freq, brk = np.histogram(x, bins=bins)
    bin_size = np.diff(brk).mean()
    x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2
    dists = {
        'Normal': stats.norm,
        'Lognorm': stats.lognorm,
        'Exponential': stats.expon,
        'Gamma': stats.gamma,
        'GEV': stats.genextreme,
        'Right-skewed Gumbel': stats.gumbel_r,
        'Generalized Pareto': stats.genpareto,
        'Pearson Type III': stats.pearson3,
    }
    df = pd.DataFrame(
        {'Dist_frozen': '', 'shape': '', 'loc': np.nan, 'scale': np.nan, 'SSE': np.nan},
        index=dists.keys()
    )
    rp = np.array([1.5, 2, 2.5, 3, 4, 5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 500])
    res_df = pd.DataFrame({'Annual return period': rp})
    for name, dist in dists.items():
        *shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
        df.at[name, 'shape'] = shape
        df.loc[name, ['loc', 'scale']] = loc, scale
        dist_frozen = dist(*shape, loc=loc, scale=scale)
        freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
        df.at[name, 'Dist_frozen'] = dist_frozen
        df.at[name, 'SSE'] = ((freq - freq_pred) ** 2).sum()
        res_df = res_df.join(pd.DataFrame({name: dist_frozen.isf(1/rp)}))
    df = df.sort_values(by='SSE', ascending=True)
    res_df = res_df.loc[:, ['Annual return period'] + df.index.tolist()]
    if graph:
        best_dist = f'{df.index[0]} distribution'
        x_plot = np.linspace(x.min(), x.max(), 100)
        y_pred = df.iloc[0, :].at['Dist_frozen'].pdf(x_plot) * bin_size * sample_size
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(x, bins=bins, density=False, alpha=.4, label='Histogram (sample)')
        ax.plot(x_plot, y_pred, 'r-.', lw=1.4, label=best_dist)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{best_dist}, SSE = {df.iloc[0].at["SSE"]:.4f}', fontsize=14)
        ax.legend()
        return fig
    return df, res_df


def fal(
        x: Union[Sequence[float], np.ndarray, pd.Series],
        bins: int = 10,
        graph: bool = True,
        figsize: Tuple[float, float] = (11, 6),
        xlabel: str = ''
    ) -> Union[mpl.figure.Figure, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    **F**requency **A**nalisys for **L**ow values (fal)

    Parameters
    ----------
    x : array-like (1d)
        One dimensional array-like of float type
    bins : int, default=10
        The number of equal-width bins in the range.
    graph : bool, default=True
        Output as a figure (True), a tuple of two DataFrames otherwise.
    figsize : tuple, default=(11, 6), in case `graph=True`
        The size of the output figure.
    xlabel : str, default=''
        The customised string for x label.

    Returns
    -------
    matplotlib.figure.Figure (`graph=True`)
        A plot with fitted distributions with sample histogram.
    A tuple of two DataFrames (`graph=False`):
        1. The DataFrame of distribution parameters with an ascending order of SSEs\n
        2. The estimated values from different annual return periods.

    Notes
    -----
    1. This function is simply used the SSE (sum of squared errors) as a fit of goodness\n
    2. The distributions (all from existing from `scipy.stats`) used:
        * 'Gamma': stats.gamma,
        * 'GEV': stats.genextreme,
        * 'Lognorm': stats.lognorm,
        * 'Pearson Type III': stats.pearson3,
        * 'Right-skewed Gumbel': stats.gumbel_r,
        * 'Weibull Minimum Extreme Value': stats.weibull_min\n
    3. In the function:
        * F(x) = P{X <= x|X >= 0} -> The cdf of X >= 0
        * F*(x) = P{X <= x|X > 0} -> The cdf of X > 0
        * 1 - F(x) = k[1 - F*(x)], where k = P{X > 0}, i.e., proportion of non-zero values

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/reference/stats.html
    """
    x = np.array(x)
    k = x[x != 0].size / x.size
    rp = np.array([1.5, 2, 3, 4, 5, 10, 20, 25, 30, 40, 50, 100])
    ref = pd.DataFrame({'Annual return period': rp})
    ref['k>='] = ref.eval('(`Annual return period` - 1) / `Annual return period`')
    ref['F(x)'] = ref.eval('1 / `Annual return period`')
    ref['F*(x)'] = ref.eval('(1 / `Annual return period` - 1 + @k) / @k')
    freq, brk = np.histogram(x, bins=bins)
    bin_size = np.diff(brk).mean()
    x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2
    tmp = ref.query('`F*(x)` > 0')
    res_df = tmp.loc[tmp.index, ['Annual return period']]
    cdf_non_zero = tmp['F*(x)'].values
    dists = {
        'Gamma': stats.gamma,
        'GEV': stats.genextreme,
        'Log Normal': stats.lognorm,
        'Pearson Type III': stats.pearson3,
        'Right-skewed Gumbel': stats.gumbel_r,
        'Weibull Minimum Extreme Value': stats.weibull_min,
    }
    df = pd.DataFrame(
        {'Dist_frozen': '', 'shape': '', 'loc': np.nan, 'scale': np.nan, 'SSE': np.nan},
        index=dists.keys()
    )
    sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
    for name, dist in dists.items():
        *shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
        df.at[name, 'shape'] = shape
        df.loc[name, ['loc', 'scale']] = loc, scale
        dist_frozen = dist(*shape, loc=loc, scale=scale)
        freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
        df.at[name, 'Dist_frozen'] = dist_frozen
        df.at[name, 'SSE'] = ((freq - freq_pred) ** 2).sum()
        res_df = res_df.join(pd.DataFrame({name: dist_frozen.ppf(cdf_non_zero)}))
    df = df.sort_values(by='SSE', ascending=True)
    res_df = res_df.loc[:, ['Annual return period'] + df.index.tolist()]
    if graph:
        colors = ('k', 'm', 'b', 'r', 'g')
        markers = ('.', '1', '2', '3', '4')
        x_plot = np.linspace(x.min(), x.max(), 100)
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(x, bins=bins, density=False, alpha=.4)
        for (i, r), color, marker in zip(df.iterrows(), colors, markers):
            y_pred = r['Dist_frozen'].pdf(x_plot) * bin_size * sample_size
            ax.plot(x_plot, y_pred, f'{marker}-.{color}', lw=.4, ms=6, alpha=.5, label=i)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Ascending order of SSE in the legend', fontsize=14)
        ax.legend()
        return fig
    return df, res_df


