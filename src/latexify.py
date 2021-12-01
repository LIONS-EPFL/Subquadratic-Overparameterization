import math
import os
import pathlib
import itertools

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


LINES = ('-', '--', ':', '-.', '-', '--')
MARKER = ('o', 's', '^', '*', '>', '<', '8', 'p')
CMAP = 'viridis'
COLORS = sns.color_palette('colorblind').as_hex()


def savefig(fig, filepath, **kwargs):
    filepath = os.path.join('figs', filepath)
    dir_name = os.path.dirname(filepath)
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True) 
    fig.savefig(filepath, **kwargs)


def latexify(fig_width=None, fig_height=None, columns=3):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.


    Code is adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples. 
    Width and max height in inches for IEEE journals taken from https://computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    assert(columns in [1,2,3,4])

    # width in inches
    if fig_width is None:
        if columns==1:
            fig_width = 6.9
        elif columns==2:
            fig_width = 3.39 
        elif columns==3:
            fig_width = 2.2
        else:
            fig_width = 1.7

    if fig_height is None:
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    fig_size = [fig_width,fig_height]

    params = {
            'backend': 'pgf',
            'pgf.preamble': [
                r'\usepackage[utf8x]{inputenc}',
                r'\usepackage[T1]{fontenc}',
                r'\usepackage{gensymb}',
            ],
            'axes.labelsize': 8, # fontsize for x and y labels (was 10)
            'axes.titlesize': 8,
            #'font.fontsize': 8, # was 10
            'legend.fontsize': 8, # was 10
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.usetex': True,
            'figure.figsize': fig_size,
            'font.family': 'sans-serif',
        
            'legend.markerscale': .9,
            'legend.numpoints': 1,
            'legend.handlelength': 2,
            'legend.scatterpoints': 1,
            'legend.labelspacing': 0.5,
            'legend.facecolor': '#eff0f1',
            # 'legend.edgecolor': 'none',
            'legend.handletextpad': 0.5,  # pad between handle and text
            'legend.borderaxespad': 0.5,  # pad between legend and axes
            'legend.borderpad': 0.5,  # pad between legend and legend content
            'legend.columnspacing': 1,  # pad between each legend column
            'axes.spines.left': True,
            'axes.spines.top': True,
            'axes.titlesize': 'medium',
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.axisbelow': True,
            'axes.grid': True,
            'grid.linewidth': 0.5,
            'grid.linestyle': '-',
            'grid.alpha': .6,
            'lines.linewidth': 1,
            'lines.markersize': 4,
            'lines.markeredgewidth': 1,

            # Force white to avoid transparent turning into black in latex.
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',

            # 'axes.prop_cycle': cycler('linestyle', LINES),

        # 'path.simplify': True,
        # 'path.simplify_threshold': 0.1,
        # 'pgf.preamble': [
        #     r'\usepackage[utf8x]{inputenc}',
        #     r'\usepackage[T1]{fontenc}',
        #     #rf'\usepackage{{{typeface}}}'
        # ],
        # Colors
    }

    matplotlib.rcParams.update(params)
    from matplotlib import rc

    sns.set_style("whitegrid")
    sns.set_palette(COLORS)

    rc('font', **{'family': 'serif', 'serif': 'cm'})
    rc('text', usetex=True)

    matplotlib.rcParams.update({
            'patch.linewidth': 0.5,
            'legend.markerscale': .9,
            'legend.numpoints': 1,
            'legend.handlelength': 2,
            'legend.scatterpoints': 1,
            'legend.labelspacing': 0.5,
            'legend.facecolor': '#ffffff',
            'legend.fancybox': False,
            # 'legend.edgecolor': 'none',
            'legend.handletextpad': 0.5,  # pad between handle and text
            'legend.borderaxespad': 0.6,  # pad between legend and axes
            'legend.borderpad': 0.6,  # pad between legend and legend content
            'legend.columnspacing': 1,  # pad between each legend column
    })

    matplotlib.rcParams.update({'image.cmap': CMAP})
