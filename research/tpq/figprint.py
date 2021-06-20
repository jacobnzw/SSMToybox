import numpy as np
import matplotlib as mpl
mpl.use('pgf')


class FigurePrint:

    INCH_PER_PT = 1.0 / 72.27  # Convert pt to inch
    PHI = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)

    def __init__(self, fig_width_pt=252):
        """

        Parameters
        ----------
        fig_width_pt : float
            Width of the figure in points, usually obtained from the journal specs or using the LaTeX command
            ``\the\columnwidth``. Default is ``fig_width_pt=252`` (3.5 inches).
        """
        self.fig_width_pt = fig_width_pt
        pgf_with_latex = {  # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "font.family": "serif",
            "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
            "font.sans-serif": [],
            "font.monospace": [],
            "font.size": 10,
            "axes.labelsize": 10,  # LaTeX default is 10pt font.
            "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # "axes.prop_cycle": ['#5DA5DA', '#FAA43A', '#60BD68',
            #                     '#F17CB0', '#B2912F', '#B276B2',
            #                     '#DECF3F', '#F15854', '#4D4D4D'],
            "figure.figsize": self.figsize(),  # default fig size
            "pgf.preamble": [  # plots will be generated using this preamble
                r"\usepackage[utf8]{inputenc}",  # use utf8 fonts
                r"\usepackage[T1]{fontenc}",
            ]
        }
        mpl.rcParams.update(pgf_with_latex)

    def figsize(self, w_scale=1.0, h_scale=1.0):
        """
        Calculates figure width and height given the width and height scale.

        Parameters
        ----------
        w_scale: float
            Figure width scale.

        h_scale: float
            Figure height scale.

        Returns
        -------
        list
            Figure width and height in inches.
        """

        fig_width = self.fig_width_pt * self.INCH_PER_PT * w_scale    # width in inches
        fig_height = fig_width * self.PHI * h_scale         # height in inches
        return [fig_width, fig_height]

    def update_default_figsize(self, fig_width_pt):
        """
        Updates default figure size used for saving.

        Parameters
        ----------
        fig_width_pt : float
            Width of the figure in points, usually obtained from the journal specs or using the LaTeX command
            ``\the\columnwidth``.


        Returns
        -------

        """
        self.fig_width_pt = fig_width_pt
        mpl.rcParams.update({"figure.figsize": self.figsize()})

    @staticmethod
    def savefig(filename):
        """
        Save figure to PGF. PDF copy created for viewing convenience.

        Parameters
        ----------
        filename

        Returns
        -------

        """
        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))

import matplotlib.pyplot as plt
