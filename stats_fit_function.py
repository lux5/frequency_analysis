
# https://nedyoxall.github.io/fitting_all_of_scipys_distributions.html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


array = stats.norm.rvs(loc=7, scale=13, size=10000, random_state=0)
bins = 100
plot_hist = True
plot_best_fit = True
plot_all_fits = False






def fit_scipy_distributions(array, bins, plot_hist=True, plot_best_fit=True, plot_all_fits=False):
    """
    Fits a range of Scipy's distributions (see scipy.stats) against an array-like input.
    Returns the sum of squared error (SSE) between the fits and the actual distribution.
    Can also choose to plot the array's histogram along with the computed fits.
    N.B. Modify the "CHANGE IF REQUIRED" comments!

    Input: array - array-like input
           bins - number of bins wanted for the histogram
           plot_hist - boolean, whether you want to show the histogram
           plot_best_fit - boolean, whether you want to overlay the plot of the best fitting distribution
           plot_all_fits - boolean, whether you want to overlay ALL the fits (can be messy!)

    Returns: results - dataframe with SSE and distribution name, in ascending order (i.e. best fit first)
             best_name - string with the name of the best fitting distribution
             best_params - list with the parameters of the best fitting distribution.
    """

    if plot_best_fit or plot_all_fits:
        assert plot_hist, "plot_hist must be True if plot_best_fit or plot_all_fits is True"

    # Returns un-normalised (i.e. counts) histogram
    y, x = np.histogram(np.array(array), bins=bins)

    # Some details about the histogram
    bin_width = x[1] - x[0]
    N = len(array)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0  # go from bin edges to bin middles

    # selection of available distributions
    # CHANGE THIS IF REQUIRED
    DISTRIBUTIONS = [
        stats.alpha,
        stats.cauchy,
        stats.cosine,
        stats.laplace,
        stats.levy,
        stats.levy_l,
        stats.norm
    ]

    if plot_hist:
        fig, ax = plt.subplots()
        h = ax.hist(np.array(array), bins=bins, color='r', alpha=.1)

    # loop through the distributions and store the sum of squared errors
    # so we know which one eventually will have the best fit
    sses = []
    for dist in DISTRIBUTIONS:
        name = dist.__class__.__name__[:-4]

        params = dist.fit(np.array(array))
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
        pdf_scaled = pdf * bin_width * N  # to go from pdf back to counts need to un-normalise the pdf

        sse = np.sum((y - pdf_scaled) ** 2)
        sses.append([sse, name])

        # Not strictly necessary to plot, but pretty patterns
        if plot_all_fits:
            ax.plot(x_mid, pdf_scaled, label=name)

    if plot_all_fits:
        plt.legend(loc=1)

    # CHANGE THIS IF REQUIRED
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')

    # Things to return - df of SSE and distribution name, the best distribution and its parameters
    results = pd.DataFrame(sses, columns=['SSE', 'distribution']).sort_values('SSE')
    best_name = results.iat[0, 1]
    best_dist = getattr(stats, best_name)
    best_params = best_dist.fit(np.array(array))

    if plot_best_fit:
        new_x = np.linspace(x_mid[0] - (bin_width * 2), x_mid[-1] + (bin_width * 2), 1000)
        best_pdf = best_dist.pdf(new_x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        best_pdf_scaled = best_pdf * bin_width * N
        ax.plot(new_x, best_pdf_scaled, label = best_name)
        plt.legend(loc=1)

    if plot_hist:
        plt.show()

    return results, best_name, best_params


DISTRIBUTIONS = [
    stats.alpha,
    stats.anglit,
    stats.arcsine,
    stats.beta,
    stats.betaprime,
    stats.bradford,
    stats.burr,
    stats.cauchy,
    stats.chi,
    stats.chi2,
    stats.cosine,
    stats.dgamma,
    stats.dweibull,
    stats.erlang,
    stats.expon,
    stats.exponnorm,
    stats.exponweib,
    stats.exponpow,
    stats.f,
    stats.fatiguelife,
    stats.fisk,
    stats.foldcauchy,
    stats.foldnorm,
#     stats.frechet_r,
#     stats.frechet_l,
    stats.genlogistic,
    stats.genpareto,
    stats.gennorm,
    stats.genexpon,
    stats.genextreme,
    stats.gausshyper,
    stats.gamma,
    stats.gengamma,
    stats.genhalflogistic,
    stats.gilbrat,
    stats.gompertz,
    stats.gumbel_r,
    stats.gumbel_l,
    stats.halfcauchy,
    stats.halflogistic,
    stats.halfnorm,
    stats.halfgennorm,
    stats.hypsecant,
    stats.invgamma,
    stats.invgauss,
    stats.invweibull,
    stats.johnsonsb,
    stats.johnsonsu,
    stats.ksone,
    stats.kstwobign,
    stats.laplace,
    stats.levy,
    stats.levy_l,
    stats.levy_stable,
    stats.logistic,
    stats.loggamma,
    stats.loglaplace,
    stats.lognorm,
    stats.lomax,
    stats.maxwell,
    stats.mielke,
    stats.nakagami,
    stats.ncx2,
    stats.ncf,
    stats.nct,
    stats.norm,
    stats.pareto,
    stats.pearson3,
    stats.powerlaw,
    stats.powerlognorm,
    stats.powernorm,
    stats.rdist,
    stats.reciprocal,
    stats.rayleigh,
    stats.rice,
    stats.recipinvgauss,
    stats.semicircular,
    stats.t,
    stats.triang,
    stats.truncexpon,
    stats.truncnorm,
    stats.tukeylambda,
    stats.uniform,
    stats.vonmises,
    stats.vonmises_line,
    stats.wald,
    stats.weibull_min,
    stats.weibull_max,
    stats.wrapcauchy
]


s = """
1924	312.1
1925	311.04
1926	310.79
1927	310.64
1928	311.25
1929	310.59
1930	310.29
1931	310.84
1932	310.33
1933	310.66
1934	310.36
1935	310.61
1936	310.56
1937	310.49
1938	310.41
1939	310.41
1940	310.89
1941	310.59
1942	310.76
1943	310.59
1944	310.36
1945	310.82
1946	311.32
1947	310.29
1948	311.33
1949	311.1
1950	310.78
1951	310.37
1952	311.1
1953	310.52
1954	310.37
1955	310.39
1956	310.75
1957	311.59
1958	311.42
1959	310.61
1960	310.72
1961	310.42
1962	310.29
1963	310.032
1964	310.635
1965	310.583
1966	310.488
1967	311.151
1968	311.232
1969	311.058
1970	311.014
1971	310.419
1972	310.681
1973	310.693
1974	310.272
1975	311.206
1976	310.383
1977	310.51
1978	311.105
1979	310.983
1980	310.735
1981	310.464
1982	311.177
1983	311.692
1984	311.256
1985	310.91
1986	310.742
1987	310.744
1988	311.376
1989	310.442
1990	310.796
1991	310.737
1992	310.578
1993	310.548
1994	311.685
1995	311.622
1996	311.427
1997	310.864
1998	310.842
1999	312.77
2000	310.834
2001	310.842
2002	310.997
2003	310.582
2004	310.612
2005	310.516
2006	310.935
2007	310.582
2008	310.471
2009	310.635
2010	311.478
2011	310.943
2012	310.389
2013	311.146
2014	310.578
2015	310.68
2016	310.754
2017	310.598
2018	310.42
2019	311.348
2020	311.326
"""

sep = '    '
ss = [[int(i.split(sep)[0]), float(i.split(sep)[1])] for i in s.split('\n')[1:-1]]
import numpy as np
import pandas as pd
df = pd.DataFrame(ss, columns=['Year', 'Y_max']).set_index('Year')
# n = len(df)

fig, ax = plt.subplots()
ax.hist(df['Y_max'], bins=10, density=True)
plt.show()

# Try to fit the gamma distribution
from scipy import stats
fit_alpha, fit_loc, fit_beta = stats.gamma.fit(df['Y_max'])
print(fit_alpha, fit_loc, fit_beta)

gamma(a=i, loc=0, scale=1)

x = np.linspace(df['Y_max'].min(), df['Y_max'].max(), 100)
m_gamma = stats.gamma(a=fit_alpha, loc=fit_loc, scale=fit_beta)
y = m_gamma.pdf(x)


fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(df['Y_max'], bins=10, density=True)
ax.plot(x, y_gamma, lw=1, color='r')
ax.set_title('Gamma probability density function fit', fontsize=14)

plt.show()







# Returns un-normalised (i.e. counts) histogram
y, x = np.histogram(np.array(array), bins=bins)
