
# https://nedyoxall.github.io/fitting_all_of_scipys_distributions.html
# https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.gamma.html


# The annual maximum lake level data from Excel
s = \
"""
1,1924,312.1,,,,,,,
2,1925,311.04,,,,,,,
3,1926,310.79,,,,,,,
4,1927,310.64,,,,,,,
5,1928,311.25,,,,,,,
6,1929,310.59,,,,,,,
7,1930,310.29,,,,,,,
8,1931,310.84,,,,,,,
9,1932,310.33,,,,,,,
10,1933,310.66,,,,,,,
11,1934,310.36,,,,,,,
12,1935,310.61,,,,,,,
13,1936,310.56,,,,,,,
14,1937,310.49,,,,,,,
15,1938,310.41,,,,,,,
16,1939,310.41,,,,,,,
17,1940,310.89,,,,,,,
18,1941,310.59,,,,,,,
19,1942,310.76,,,,,,,
20,1943,310.59,,,,,,,
21,1944,310.36,,,,,,,
22,1945,310.82,,,,,,,
23,1946,311.32,,,,,,,
24,1947,310.29,,,,,,,
25,1948,311.33,,,,,,,
26,1949,311.1,,,,,,,
27,1950,310.78,,,,,,,
28,1951,310.37,,,,,,,
29,1952,311.1,,,,,,,
30,1953,310.52,,,,,,,
31,1954,310.37,,,,,,,
32,1955,310.39,,,,,,,
33,1956,310.75,,,,,,,
34,1957,311.59,,,,,,,
35,1958,311.42,,,,,,,
36,1959,310.61,,,,,,,
37,1960,310.72,,,,,,,
38,1961,310.42,,,,,,,
39,1962,310.29,,,,,,,
40,1963,310.032,,,,,,,
41,1964,310.635,,,,,,,
42,1965,310.583,,,,,,,
43,1966,310.488,,,,,,,
44,1967,311.151,,,,,,,
45,1968,311.232,,,,,,,
46,1969,311.058,,,,,,,
47,1970,311.014,,,,,,,
48,1971,310.419,,,,,,,
49,1972,310.681,,,,,,,
50,1973,310.693,,,,,,,
51,1974,310.272,,,,,,,
52,1975,311.206,,,,,,,
53,1976,310.383,,,,,,,
54,1977,310.51,,,,,,,
55,1978,311.105,,,,,,,
56,1979,310.983,,,,,,,
57,1980,310.735,,,,,,,
58,1981,310.464,,,,,,,
59,1982,311.177,,,,,,,
60,1983,311.692,,,,,,,
61,1984,311.256,,,,,,,
62,1985,310.91,,,,,,,
63,1986,310.742,,,,,,,
64,1987,310.744,,,,,,,
65,1988,311.376,,,,,,,
66,1989,310.442,,,,,,,
67,1990,310.796,,,,,,,
68,1991,310.737,,,,,,,
69,1992,310.578,,,,,,,
70,1993,310.548,,,,,,,
71,1994,311.685,,,,,,,
72,1995,311.622,,,,,,,
73,1996,311.427,,,,,,,
74,1997,310.864,,,,,,,
75,1998,310.842,,,,,,,
76,1999,312.77,,,,,,,
77,2000,310.834,,,,,,,
78,2001,310.842,,,,,,,
79,2002,310.997,,,,,,,
80,2003,310.582,,,,,,,
81,2004,310.612,,,,,,,
82,2005,310.516,,,,,,,
83,2006,310.935,,,,,,,
84,2007,310.582,,,,,,,
85,2008,310.471,,,,,,,
86,2009,310.635,,,,,,,
87,2010,311.478,,,,,,,
88,2011,310.943,,,,,,,
89,2012,310.389,,,,,,,
90,2013,311.146,,,,,,,
91,2014,310.578,,,,,,,
92,2015,310.68,,,,,,,
93,2016,310.754,,,,,,,
94,2017,310.598,,,,,,,
95,2018,310.42,,,,,,,
96,2019,311.348,,,,,,,
97,2020,311.326,,,,,,,
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Apply the above matplotlib settings
from fun import dict_rc
for k, v in dict_rc.items():
    plt.rcParams[k] = v
del dict_rc

pd.set_option('display.max_columns', None)

# Convert the above string as a numpy array
ss = [i.split(',') for i in s.split('\n')[1:-1]]
x = np.array([float(i[2]) for i in ss])

# Get the sample size, guessed initial values for the proposed distribution fit
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()

# Calculate the histogram
bins = 10
freq, brk = np.histogram(x, bins=bins)
bin_size = np.diff(brk).mean()
x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2

# Create a series for plot
x_plot = np.linspace(x.min(), x.max(), 100)

# Target return periods
return_period = np.array([1.5, 2, 3, 4, 5, 10, 20, 25, 50, 100, 200])

# %%

bins = 10
graph = False
figsize = (11, 6)
xlabel = 'Lake level (m)'

def fah(x, bins: int = 10, graph: bool = True, figsize: tuple = (11, 6), xlabel: str = ''):
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

    # Sample size and choose the initial guess values for loc and scale estimations
    sample_size, init_loc, init_scale = x.size, x.mean(), x.std()

    # Get histogram (frequency) values
    freq, brk = np.histogram(x, bins=bins)
    bin_size = np.diff(brk).mean()
    x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2

    # List the distributions (all from `scipy.stats`) used in this function
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

    # List of requested values under different return periods
    rp = np.array([1.5, 2, 2.5, 3, 4, 5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 500])
    res_df = pd.DataFrame({'Annual return period': rp})

    # Fit distributions (params estimations)
    for name, dist in dists.items():
        # Fit the distribution
        *shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)

        df.at[name, 'shape'] = shape
        df.loc[name, 'loc':'scale'] = loc, scale

        # Calculate the modelled frequency from the derived distribution (frozen)
        dist_frozen = dist(*shape, loc=loc, scale=scale)
        freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size

        df.at[name, 'Dist_frozen'] = dist_frozen

        # Calculate the distribution-modelled SSE
        df.at[name, 'SSE'] = ((freq - freq_pred) ** 2).sum()

        res_df = res_df.join(pd.DataFrame({name: dist_frozen.isf(1/rp)}))

    # Order the distribution by SSE in ascending order
    df = df.sort_values(by='SSE', ascending=True)
    res_df = res_df.loc[:, ['Annual return period'] + df.index.tolist()]


    # Make a plot (when `graph=True`) for the 'best' fit only
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


fah(x, bins=11, xlabel=xlabel)
plt.show()

a, b = fah(x, bins=11, graph=False)
print(a)
print('\n----\n')
print(b)

# Calculate the lake levels of return periods of 30 and 40 years
a['Dist_frozen'].apply(lambda x: x.isf(1/np.array([30, 40])))

# %%

x = np.array([float(i[2]) for i in ss])
# Get the sample size, guessed initial values for the proposed distribution fit
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
# Calculate the histogram
freq, brk = np.histogram(x, bins=bins)
bin_size = np.diff(brk).mean()
x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2
# Create a series for plot
x_plot = np.linspace(x.min(), x.max(), 100)




dist = stats.norm
name_dist = 'Normal distribution'


                # +++++++++++++++++++++++++++++++++++ #
                # --- Fit the Normal distribution --- #
                # +++++++++++++++++++++++++++++++++++ #


*shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
print(*shape, loc, scale)


# Calculate the normal-modelled SSE
dist_frozen = dist(*shape, loc=loc, scale=scale)
freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
sse = ((freq - freq_pred) ** 2).sum()


# Make a plot of observed histogram and the selected pdf
y_pred = dist_frozen.pdf(x_plot) * bin_size * sample_size

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(x, bins=bins, alpha=.4, label='Observed frequency')
ax.plot(x_plot, y_pred, 'r-.', lw=1, label=name_dist)
ax.set_xlabel('Lake Wakatipu level (m)')
ax.set_ylabel('Frequency')
ax.set_title(f'SSE = {sse:.4f}', fontsize=14)
ax.legend()
plt.show()


# Calculate the designed lake levels for different annual return periods
res_df = pd.DataFrame(
    {
        'Annual return period': return_period,
        'Designed lake level (m)': dist_frozen.isf(1/return_period)
    }
)
print(res_df)


# %%

x = np.array([float(i[2]) for i in ss])
# Get the sample size, guessed initial values for the proposed distribution fit
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
# Calculate the histogram
freq, brk = np.histogram(x, bins=bins)
bin_size = np.diff(brk).mean()
x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2
# Create a series for plot
x_plot = np.linspace(x.min(), x.max(), 100)




dist = stats.lognorm
name_dist = 'Log Normal distribution'


                # ++++++++++++++++++++++++++++++++++++++++++++++++ #
                # --- Fit the Log-normal distribution (direct) --- #
                # ++++++++++++++++++++++++++++++++++++++++++++++++ #


*shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
print(*shape, loc, scale)


# Calculate the normal-modelled SSE
dist_frozen = dist(*shape, loc=loc, scale=scale)
freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
sse = ((freq - freq_pred) ** 2).sum()


# Make a plot of observed histogram and the selected pdf
y_pred = dist_frozen.pdf(x_plot) * bin_size * sample_size

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(x, bins=bins, alpha=.4, label='Observed frequency')
ax.plot(x_plot, y_pred, 'r-.', lw=1, label=name_dist)
ax.set_xlabel('Lake Wakatipu level (m)')
ax.set_ylabel('Frequency')
ax.set_title(f'SSE = {sse:.4f}', fontsize=14)
ax.legend()
plt.show()


# Calculate the designed lake levels for different annual return periods
res_df = pd.DataFrame(
    {
        'Annual return period': return_period,
        'Designed lake level (m)': dist_frozen.isf(1/return_period)
    }
)
print(res_df)

# %%

x = np.log10([float(i[2]) for i in ss])
# Get the sample size, guessed initial values for the proposed distribution fit
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
# Calculate the histogram
freq, brk = np.histogram(x, bins=bins)
bin_size = np.diff(brk).mean()
x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2
# Create a series for plot
x_plot = np.linspace(x.min(), x.max(), 100)




dist = stats.norm
name_dist = 'Log Normal distribution (indirect)'


                # ++++++++++++++++++++++++++++++++++++++++++++++++++ #
                # --- Fit the Log-normal distribution (indirect) --- #
                # ++++++++++++++++++++++++++++++++++++++++++++++++++ #


*shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
print(*shape, loc, scale)


# Calculate the normal-modelled SSE
dist_frozen = dist(*shape, loc=loc, scale=scale)
freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
sse = ((freq - freq_pred) ** 2).sum()


# Make a plot of observed histogram and the selected pdf
y_pred = dist_frozen.pdf(x_plot) * bin_size * sample_size

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(10**x, bins=bins, alpha=.4, label='Observed frequency')
ax.plot(10**x_plot, y_pred, 'r-.', lw=1, label=name_dist)
ax.set_xlabel('Lake Wakatipu level (m)')
ax.set_ylabel('Frequency')
ax.set_title(f'SSE = {sse:.4f}', fontsize=14)
ax.legend()
plt.show()


# Calculate the designed lake levels for different annual return periods
res_df = pd.DataFrame(
    {
        'Annual return period': return_period,
        'Designed lake level (m)': 10**dist_frozen.isf(1/return_period)
    }
)
print(res_df)












# %%

x = np.array([float(i[2]) for i in ss])
# Get the sample size, guessed initial values for the proposed distribution fit
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
# Calculate the histogram
freq, brk = np.histogram(x, bins=bins)
bin_size = np.diff(brk).mean()
x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2
# Create a series for plot
x_plot = np.linspace(x.min(), x.max(), 100)




dist = stats.pearson3
name_dist = 'Pearson Type III distribution'


                # ++++++++++++++++++++++++++++++++++++++++++++++++++ #
                # --- Fit the Pearson3 Distribution distribution --- #
                # ++++++++++++++++++++++++++++++++++++++++++++++++++ #


*shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
print(*shape, loc, scale)


# Calculate the normal-modelled SSE
dist_frozen = dist(*shape, loc=loc, scale=scale)
freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
sse = ((freq - freq_pred) ** 2).sum()


# Make a plot of observed histogram and the selected pdf
y_pred = dist_frozen.pdf(x_plot) * bin_size * sample_size

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(x, bins=bins, alpha=.4, label='Observed frequency')
ax.plot(x_plot, y_pred, 'r-.', lw=1, label=name_dist)
ax.set_xlabel('Lake Wakatipu level (m)')
ax.set_ylabel('Frequency')
ax.set_title(f'SSE = {sse:.4f}', fontsize=14)
ax.legend()
plt.show()


# Calculate the designed lake levels for different annual return periods
res_df = pd.DataFrame(
    {
        'Annual return period': return_period,
        'Designed lake level (m)': dist_frozen.isf(1/return_period)
    }
)
print(res_df)


# %%

x = np.log10([float(i[2]) for i in ss])
# Get the sample size, guessed initial values for the proposed distribution fit
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
# Calculate the histogram
freq, brk = np.histogram(x, bins=bins)
bin_size = np.diff(brk).mean()
x_mid = (brk + np.roll(brk, shift=-1))[:-1] / 2
# Create a series for plot
x_plot = np.linspace(x.min(), x.max(), 100)





dist = stats.pearson3
name_dist = 'Log Pearson Type III distribution'


                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
                # --- Fit the Log Pearson3 Distribution distribution --- #
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


*shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
print(*shape, loc, scale)


# Calculate the normal-modelled SSE
dist_frozen = dist(*shape, loc=loc, scale=scale)
freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
sse = ((freq - freq_pred) ** 2).sum()


# Make a plot of observed histogram and the selected pdf
y_pred = dist_frozen.pdf(x_plot) * bin_size * sample_size

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(10**x, bins=bins, alpha=.4, label='Observed frequency')
ax.plot(10**x_plot, y_pred, 'r-.', lw=1, label=name_dist)
ax.set_xlabel('Lake Wakatipu level (m)')
ax.set_ylabel('Frequency')
ax.set_title(f'SSE = {sse:.4f}', fontsize=14)
ax.legend()
plt.show()


# Calculate the designed lake levels for different annual return periods
res_df = pd.DataFrame(
    {
        'Annual return period': return_period,
        'Designed lake level (m)': 10**dist_frozen.isf(1/return_period)
    }
)
print(res_df)

# %%


