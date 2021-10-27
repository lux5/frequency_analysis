
import matplotlib.pyplot as plt

# Matplotlib settings (most from proplot package)
dict_rc = {
    'axes.grid': True,
    'axes.linewidth': 0.6,
    'axes.titlepad': 5.0,
    'errorbar.capsize': 3.0,
    'figure.figsize': [4, 4],
    'figure.facecolor': 'f4f4f4',
    'figure.titleweight': 'bold',
    'font.size': 9.0,
    'grid.alpha': 0.1,
    'grid.color': 'black',
    'grid.linewidth': 0.6,
    'hatch.linewidth': 0.6,
    'legend.borderaxespad': 0,
    'legend.borderpad': 0.5,
    'legend.columnspacing': 1.5,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.fancybox': False,
    'legend.handletextpad': 0.5,
    'mathtext.fontset': 'custom',
    'mathtext.default': 'regular',
    'patch.linewidth': 0.6,
    'savefig.dpi': 140,
    'savefig.facecolor': 'white',
    'xtick.major.pad': 2.0,
    'xtick.major.size': 4.0,
    'xtick.major.width': 0.6,
    'xtick.minor.pad': 2.0,
    'xtick.minor.width': 0.48,
    'xtick.minor.visible': True,
    'ytick.major.pad': 2.0,
    'ytick.major.size': 4.0,
    'ytick.major.width': 0.6,
    'ytick.minor.pad': 2.0,
    'ytick.minor.width': 0.48,
    'ytick.minor.visible': True,
    'axes.titlecolor': 'black',
}

# Apply the above matplotlib settings
for k, v in dict_rc.items():
    plt.rcParams[k] = v

