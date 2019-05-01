"""Create and save figures from the saved exemplar models' accuracies."""
from __future__ import print_function
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from shapes_experiment.misc import (FIGURES_DIR, ACCURACY_DIR,
                                    create_and_save_figure)
sns.set(style="whitegrid")
sns.set(font_scale=1.6, style="ticks")
if __name__ == '__main__':
    # Figure for exemplar_model_circle_square.py's results:
    accuracy_colour = np.loadtxt(
        ACCURACY_DIR + 'circle_square_colour_luce.csv')
    accuracy_grayscale = np.loadtxt(
        ACCURACY_DIR + 'circle_square_grayscale_luce.csv')
    filename = 'luce_choice_accuracy.pdf'

    create_and_save_figure(accuracy_colour, accuracy_grayscale, filename)

    # Figure for exemplar_model_circle_square_bounding_box.py's results:
    accuracy_colour = np.loadtxt(
        ACCURACY_DIR + 'circle_square_bounding_box_colour_luce.csv')
    accuracy_grayscale = np.loadtxt(
        ACCURACY_DIR + 'circle_square_bounding_box_grayscale_luce.csv')
    filename = 'luce_choice_accuracy_bounding_box.pdf'

    create_and_save_figure(accuracy_colour, accuracy_grayscale, filename)

    # Same hue, colour, shape, etc.
    df = pd.read_csv(ACCURACY_DIR +
                     'circle_square_bounding_box_colour_accuracy.csv')
    accuracy_shape = df['Same Shape Luce Accuracy']
    accuracy_hue = df['Same Hue Luce Accuracy']
    accuracy_size = df['Same Size Luce Accuracy']
    filename = 'luce_shape_hue_size_accuracy_bounding_box.pdf'
    # y_min = 0.49
    # y_max = 1.005

    fig = plt.figure()

    plt.plot(accuracy_hue, label='Hue', color='#33a02c', alpha=0.75,
             linewidth=3)
    plt.plot(accuracy_size, label='Size', color='#fb9a99', alpha=0.75,
             linewidth=3)
    plt.plot(accuracy_shape, label='Shape', color='#1f78b4', alpha=0.75,
             linewidth=3)
    # plt.axis([-1, 26, y_min, y_max])
    sns.despine(offset=10, trim=True)

    plt.legend(loc=0)
    plt.xlabel('Network Layer', size=20)
    plt.ylabel('Accuracy', size=20)

    plot = fig.add_subplot(111)
    plot.tick_params(axis='both', which='major', labelsize=16)
    plot.tick_params(axis='both', which='minor', labelsize=20)
    fig.savefig(FIGURES_DIR + filename,
                format='pdf', bbox_inches='tight')

    # Same hue, colour, shape, etc.
    df = pd.read_csv(ACCURACY_DIR + 'circle_square_colour_accuracy.csv')
    accuracy_shape = df['Same Shape Luce Accuracy']
    accuracy_hue = df['Same Hue Luce Accuracy']
    accuracy_size = df['Same Size Luce Accuracy']
    filename = 'luce_shape_hue_size_accuracy.pdf'
    # y_min = 0.49
    # y_max = 1.005

    fig = plt.figure()

    plt.plot(accuracy_hue, label='Hue', color='#33a02c', alpha=0.75,
             linewidth=3)
    plt.plot(accuracy_size, label='Size', color='#fb9a99', alpha=0.75,
             linewidth=3)
    plt.plot(accuracy_shape, label='Shape', color='#1f78b4', alpha=0.75,
             linewidth=3)
    # plt.axis([-1, 26, y_min, y_max])
    sns.despine(offset=10, trim=True)

    plt.legend(loc=0)
    plt.xlabel('Network Layer', size=20)
    plt.ylabel('Accuracy', size=20)

    plot = fig.add_subplot(111)
    plot.tick_params(axis='both', which='major', labelsize=16)
    plot.tick_params(axis='both', which='minor', labelsize=20)
    fig.savefig(FIGURES_DIR + filename,
                format='pdf', bbox_inches='tight')
