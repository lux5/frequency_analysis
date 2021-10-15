# iwaponline.com/ws/article/19/1/30/39335/Frequency-analysis-of-low-flows-in-intermittent

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

pd.set_option('display.max_columns', None)

from fun import dict_rc, malf
for k, v in dict_rc.items():
    plt.rcParams[k] = v


# Read the naturalised daily flow for CMB
flo = pd.read_pickle('data/roddy_nat_cmb.xz')

# Calculate the 7dLF each water year
x = malf(flo, detail=True)['7dLF'].values

# The description of the sample histogram
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
bins = 12
freq, brks = np.histogram(x, bins=bins)
bin_size = np.diff(brks).mean()
x_mid = (brks + np.roll(brks, -1))[:-1] / 2

# The target annual return periods
T = np.array([2, 5, 10, 25, 50, 100])

# Calculate the fraction of non-zero values i.e., P(X != 0)
k = x[x != 0].size / x.size

# To make the value of the CDF i.e., [F*(x) = P(X <= x|X != 0)] >= 0
k_df = pd.DataFrame({'T': return_period})
k_df['k >='] = k_df.eval('(T - 1) / T')

# k is valid for a return period that satisfies
k_df['k >=']

"""
Fit the following distributions:

    * Gamma
    * GEV
    * Log Normal
    * Pearson Type III
    * Weibull Minimum Extreme Value

"""


# %%

# Try Log Normal distribution
dist = stats.genextreme
name_dist = 'GEV'

# Fit the model
*shape, loc, scale = dist.fit(data=x, loc=init_loc, scale=init_scale)
print(*shape, loc, scale)

# Calculate the normal-modelled SSE
dist_frozen = dist(*shape, loc=loc, scale=scale)
freq_pred = dist_frozen.pdf(x_mid) * bin_size * sample_size
sse = ((freq - freq_pred) ** 2).sum()

# Create some data for simulating the frequency from the selected distribution
x_plot = np.linspace(x.min(), x.max(), 100)
y_pred = dist_frozen.pdf(x_plot) * bin_size * sample_size

# Make a plot of observed histogram and the selected pdf
fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(x, bins=bins, alpha=.4, label='Histogram (sample)')
ax.plot(x_plot, y_pred, 'r-.', lw=1, label=name_dist)
ax.set_xlabel('7dLF ($m^3/s$)')
ax.set_ylabel('Frequency')
ax.set_title(f'SSE = {sse:.4f}', fontsize=14)
ax.legend()
plt.show()

# Calculate the Q_{7, return period} for the target return periods


# %% Complete version with notes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

pd.set_option('display.max_columns', None)

from fun import dict_rc, malf
for k, v in dict_rc.items():
    plt.rcParams[k] = v


# Read the naturalised daily flow for CMB
flo = pd.read_pickle('data/roddy_nat_cmb.xz')

# Calculate the 7dLF each water year
x = malf(flo, detail=True)['7dLF'].values

# The description of the sample histogram
sample_size, init_loc, init_scale = x.size, x.mean(), x.std()
bins = 12
freq, brks = np.histogram(x, bins=bins)
bin_size = np.diff(brks).mean()
x_mid = (brks + np.roll(brks, -1))[:-1] / 2

# The target annual return periods (T)
T = np.array([2, 5, 10, 25, 50, 100])

# Calculate the fraction of non-zero values i.e., [P(X != 0)]
k = x[x != 0].size / x.size

#



