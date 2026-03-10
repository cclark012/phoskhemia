from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

nodes: list[float] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

colors: list[str] = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
cb2_purples = LinearSegmentedColormap.from_list('cb2_purples', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=cb2_purples)

colors: list[str] = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
cb2_blues = LinearSegmentedColormap.from_list('cb2_blues', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=cb2_blues)

colors: list[str] =  ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
cb2_greens = LinearSegmentedColormap.from_list('cb2_greens', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=cb2_greens)

colors: list[str] = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
cb2_oranges = LinearSegmentedColormap.from_list('cb2_oranges', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=cb2_oranges)

colors: list[str] = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
cb2_reds = LinearSegmentedColormap.from_list('cb2_reds', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=cb2_reds)

colors: list[str] = ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000']
cb2_grays = LinearSegmentedColormap.from_list('cb2_grays', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=cb2_grays)

def plot_linearmap(cmap):
    rgba = cmap(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')

    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])

    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    plt.show()
