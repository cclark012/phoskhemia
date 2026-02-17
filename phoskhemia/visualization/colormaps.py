from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

nodes = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

colors = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
purples = LinearSegmentedColormap.from_list('purples', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=purples)

colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
blues = LinearSegmentedColormap.from_list('blues', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=blues)

colors =  ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
greens = LinearSegmentedColormap.from_list('greens', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=greens)

colors = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
oranges = LinearSegmentedColormap.from_list('oranges', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=oranges)

colors = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
reds = LinearSegmentedColormap.from_list('reds', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=reds)

colors = ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000']
grays = LinearSegmentedColormap.from_list('grays', list(zip(nodes, colors)), N=256)
mpl.colormaps.register(cmap=grays)

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
