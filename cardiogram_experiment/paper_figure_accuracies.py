"""Graph the accuracies for exemplar models."""
from __future__ import division, print_function

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.lines import Line2D

from cardiogram_experiment.misc import EXP_DIR, FIGURES_DIR

# sns.set(style="whitegrid")
# sns.set(font_scale=1.5, style="ticks")

def create_and_save_figures():
    sns.set()
    sns.set(style="whitegrid")
    sns.set(font_scale=1.5, style="ticks")

    ACCURACY_TYPES = ['luce', 'optimum']
    fig, axes = plt.subplots(2, 1, sharex='col', figsize=(5.5, 10))
    lw = 2
    alpha = 0.7
    for accuracy_type, ax in zip(ACCURACY_TYPES, axes):
        train_colour_test_colour = \
            np.loadtxt(EXP_DIR
                       + '/exemplar_model_train_colour_test_colour'
                       '/' + accuracy_type
                       + '_accuracy_train_colour_test_colour.csv')
        train_grayscale_test_grayscale = \
            np.loadtxt(EXP_DIR
                       + '/exemplar_model_train_grayscale_test_grayscale'
                       '/' + accuracy_type
                       + '_accuracy_train_grayscale_test_grayscale.csv')
        train_colour_test_grayscale = \
            np.loadtxt(EXP_DIR
                       + '/exemplar_model_train_colour_test_grayscale'
                       '/' + accuracy_type
                       + '_accuracy_train_colour_test_grayscale.csv')
        ax.plot(train_colour_test_colour,
                label='Train colour; test colour', color='#c63d92',
                lw=lw, alpha=alpha)
        ax.plot(train_colour_test_grayscale,
                label='Train colour; test grayscale', color='#EEA9E1',
                lw=lw, alpha=alpha)
        ax.plot(train_grayscale_test_grayscale,
                label='Train grayscale; test grayscale', color='#888888',
                lw=lw, alpha=alpha)

        if accuracy_type == 'luce':
            accuracy_type = 'γ = 1'
        else:
            accuracy_type = accuracy_type.capitalize()
        ax.set_ylabel(accuracy_type + ' Accuracy')

        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True)  # labels along the bottom edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=True,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
        )

    custom_lines = [Line2D([0], [0], color='#c63d92', lw=2),

                    Line2D([0], [0], color='#888888', lw=2),
                    Line2D([0], [0], color='#EEA9E1', lw=2), ]
    axes[0].legend(custom_lines, ['Train: color; test: color',
                                  'Train: grayscale; test: grayscale',
                                  'Train: color; test: grayscale'
                                  ],
                   loc='upper left', frameon=False)


    axes[0].annotate("a", xy=(-0.3, 0.9), xycoords="axes fraction", weight='bold')
    axes[0].tick_params(axis='both', which='both', top='off', right='off')

    axes[1].annotate("b", xy=(-0.3, 1.05), xycoords="axes fraction", weight='bold')
    axes[1].tick_params(axis='both', which='both', top='off', right='off')

    axes[0].set_ylim([0.5, 0.535])
    axes[0].set_yticks(np.arange(0.5, 0.53, 0.005))

    sns.despine(offset=10, trim=True)


    fig.tight_layout()
    axes[1].set_xlabel('Network Layer')
    fig.savefig(FIGURES_DIR + 'accuracy.pdf',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIGURES_DIR + 'accuracy.png',
                bbox_inches='tight', pad_inches=0.05)

    return FIGURES_DIR + 'accuracy.pdf'


if __name__ == '__main__':
    create_and_save_figures()
