"""Create and save figures from the saved exemplar models' accuracies."""
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from shapes_experiment.misc import (FIGURES_DIR, ACCURACY_DIR,
                                    create_and_save_figure)


def create_and_save_figures():
    filename = 'luce_accuracies'

    sns.set(style="whitegrid")
    sns.set(style="ticks")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, sharex='col', sharey='row')
    ax1.set_title('Overlapping\n')
    ax2.set_title('Non-overlapping\n')

    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_colour_accuracy.csv')
    accuracy_colour = df['Luce Accuracy']
    ax1.set_title('a', loc='left', weight='bold')
    ax1.plot(accuracy_colour, label='Color',
             color='#c63d92', linewidth=2, alpha=0.9)
    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_grayscale_accuracy.csv')
    accuracy_grayscale = df['Luce Accuracy']
    ax1.plot(accuracy_grayscale, label='Grayscale',
             color='#999999', linewidth=2, alpha=0.6)
    ax1.legend(loc='upper center', frameon=False)
    ax1.yaxis.set_ticks(np.arange(0.48, 0.62, 0.02))

    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_bounding_box_colour_accuracy.csv')
    accuracy_colour = df['Luce Accuracy']
    ax2.set_title('b', loc='left', weight='bold')
    ax2.plot(accuracy_colour, label='Color',
             color='#c63d92', linewidth=2, alpha=0.9)
    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_bounding_box_grayscale_accuracy.csv')
    accuracy_grayscale = df['Luce Accuracy']
    ax2.plot(accuracy_grayscale, label='Grayscale',
             color='#999999', linewidth=2, alpha=0.6)
    ax2.legend(loc='upper center', frameon=False)

    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_colour_accuracy.csv')
    accuracy_shape = df['Same Shape Luce Accuracy']
    accuracy_hue = df['Same Hue Luce Accuracy']
    accuracy_size = df['Same Size Luce Accuracy']
    ax3.set_title('c', loc='left', weight='bold')
    ax3.plot(accuracy_hue, label='Hue', color='#33a02c', alpha=0.75,
             linewidth=2)
    ax3.plot(accuracy_size, label='Size', color='#fb9a99', alpha=0.75,
             linewidth=2)
    ax3.plot(accuracy_shape, label='Shape', color='#1f78b4', alpha=0.75,
             linewidth=2)
    ax3.legend(loc='upper center', frameon=False)
    ax3.yaxis.set_ticks(np.arange(0.45, 0.7, 0.05))


    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_bounding_box_colour_accuracy.csv')
    accuracy_shape = df['Same Shape Luce Accuracy']
    accuracy_hue = df['Same Hue Luce Accuracy']
    accuracy_size = df['Same Size Luce Accuracy']
    ax4.set_title('d', loc='left', weight='bold')
    ax4.plot(accuracy_hue, label='Hue', color='#33a02c', alpha=0.75,
             linewidth=2)
    ax4.plot(accuracy_size, label='Size', color='#fb9a99', alpha=0.75,
             linewidth=2)
    ax4.plot(accuracy_shape, label='Shape', color='#1f78b4', alpha=0.75,
             linewidth=2)
    ax4.legend(loc='upper center', frameon=False)

    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_grayscale_accuracy.csv')
    accuracy_shape = df['Same Shape Luce Accuracy']
    accuracy_hue = df['Same Hue Luce Accuracy']
    accuracy_size = df['Same Size Luce Accuracy']
    ax5.set_title('e', loc='left', weight='bold')
    ax5.plot(accuracy_hue, label='Lightness', color='#33a02c', alpha=0.75,
             linewidth=2)
    ax5.plot(accuracy_size, label='Size', color='#fb9a99', alpha=0.75,
             linewidth=2)
    ax5.plot(accuracy_shape, label='Shape', color='#1f78b4', alpha=0.75,
             linewidth=2)
    ax5.legend(loc='upper center', frameon=False)
    ax5.yaxis.set_ticks(np.arange(0.45, 0.7, 0.05))
    ax5.set_ylim(0.45, 0.7)

    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_bounding_box_grayscale_accuracy.csv')
    accuracy_shape = df['Same Shape Luce Accuracy']
    accuracy_hue = df['Same Hue Luce Accuracy']
    accuracy_size = df['Same Size Luce Accuracy']
    ax6.set_title('f', loc='left', weight='bold')
    ax6.plot(accuracy_hue, label='Lightness', color='#33a02c', alpha=0.75,
             linewidth=2)
    ax6.plot(accuracy_size, label='Size', color='#fb9a99', alpha=0.75,
             linewidth=2)
    ax6.plot(accuracy_shape, label='Shape', color='#1f78b4', alpha=0.75,
             linewidth=2)
    ax6.legend(loc='upper center', frameon=False)

    fig.text(0.5, -0.01, 'Network Layer', ha='center', fontsize='12')
    fig.text(-0.01, 0.5, 'Accuracy', va='center',
             rotation='vertical', fontsize='12')


    ax1.tick_params(width=0.75)
    ax2.tick_params(width=0.75)
    ax3.tick_params(width=0.75)
    ax4.tick_params(width=0.75)
    ax5.tick_params(width=0.75)
    ax6.tick_params(width=0.75)

    ax1.tick_params(axis='both', which='both', top='off', right='off')
    ax2.tick_params(axis='both', which='both', top='off', right='off')
    ax3.tick_params(axis='both', which='both', top='off', right='off')
    ax4.tick_params(axis='both', which='both', top='off', right='off')
    ax5.tick_params(axis='both', which='both', top='off', right='off')
    ax6.tick_params(axis='both', which='both', top='off', right='off')

    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(0.75)
      ax2.spines[axis].set_linewidth(0.75)
      ax3.spines[axis].set_linewidth(0.75)
      ax4.spines[axis].set_linewidth(0.75)
      ax5.spines[axis].set_linewidth(0.75)
      ax6.spines[axis].set_linewidth(0.75)




    # sns.despine(offset={'left':10,'right':0,'top':0,'bottom':0}, trim=True)
    sns.despine(offset=10, trim=True)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"


    fig.savefig(FIGURES_DIR + filename + '.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR + filename + '.png', bbox_inches='tight')
    return FIGURES_DIR + filename + '.pdf'


if __name__ == '__main__':
    create_and_save_figures()
