"""Graph the accuracies for all exemplar models."""
from __future__ import division, print_function

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from cardiogram_experiment.misc import EXP_DIR, FIGURES_DIR

sns.set(style="whitegrid")
sns.set(font_scale=1.8, style="ticks")

ACCURACY_TYPES = ['luce', 'optimum']
LOCS = ['upper left', 'lower left']

for a, loc in zip(ACCURACY_TYPES, LOCS):
    fig = plt.figure()
    train_colour_test_colour = \
        np.loadtxt(EXP_DIR +
                   '/exemplar_model_train_colour_test_colour'
                   '/' + a + '_accuracy_train_colour_test_colour.csv')
    train_gray_test_gray = \
        np.loadtxt(EXP_DIR +
                   '/exemplar_model_train_gray_test_gray'
                   '/' + a + '_accuracy_train_gray_test_gray.csv')
    train_colour_test_gray = \
        np.loadtxt(EXP_DIR +
                   '/exemplar_model_train_colour_test_gray'
                   '/' + a + '_accuracy_train_colour_test_gray.csv')
    plt.plot(train_colour_test_colour,
             label='Train colour; test colour')
    plt.plot(train_gray_test_gray, label='Train gray; test gray')
    plt.plot(train_colour_test_gray, label='Train colour; test gray')
    plt.xlabel('Network Layer')
    plt.ylabel(a.title() + ' Choice Accuracy')
    plt.legend(frameon=False, loc=loc, borderaxespad=0.)
    sns.despine(trim=True)
    fig.savefig(FIGURES_DIR + a +
                '_accuracy_as_function_of_train_test_gray.pdf',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIGURES_DIR + a +
                '_accuracy_as_function_of_train_test_gray.png',
                bbox_inches='tight', pad_inches=0.05)

    fig = plt.figure()
    train_colour_test_colour = \
        np.loadtxt(EXP_DIR +
                   '/exemplar_model_train_colour_test_colour'
                   '/' + a + '_accuracy_train_colour_test_colour.csv')
    train_grayscale_test_grayscale = \
        np.loadtxt(EXP_DIR +
                   '/exemplar_model_train_grayscale_test_grayscale'
                   '/' + a + '_accuracy_train_grayscale_test_grayscale.csv')
    train_colour_test_grayscale = \
        np.loadtxt(EXP_DIR +
                   '/exemplar_model_train_colour_test_grayscale'
                   '/' + a + '_accuracy_train_colour_test_grayscale.csv')
    plt.plot(train_colour_test_colour,
             label='Train colour; test colour')
    plt.plot(train_grayscale_test_grayscale,
             label='Train grayscale; test grayscale')
    plt.plot(train_colour_test_grayscale,
             label='Train colour; test grayscale')
    plt.xlabel('Network Layer')
    plt.ylabel(a.title() + ' Choice Accuracy')
    plt.legend(frameon=False, loc=loc, borderaxespad=0.)
    # sns.despine(trim=True)
    sns.despine(offset=10, trim=True)

    fig.savefig(FIGURES_DIR + a + '_accuracy_as_function_of_train_test.pdf',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIGURES_DIR + a + '_accuracy_as_function_of_train_test.png',
                bbox_inches='tight', pad_inches=0.05)
