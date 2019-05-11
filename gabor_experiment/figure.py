from __future__ import division, print_function

import os

import numpy as np
import pandas as pd
import seaborn as sns

from gabor_experiment.misc import FIGS_DIRS
from gabor_experiment.misc import REPS_DIRS
from gabor_experiment.misc import RESULTS_DIR
from gabor_experiment.misc import STIMULI_DIRS
from gabor_experiment.misc import STIMULI_POSTFIXES
from gabor_experiment.misc import TITLES
import matplotlib.pyplot as plt
from utils.misc import LAYER_NAMES


def create_and_save_figures(just_original=True):
    """Create and save figures."""
    sns.set()
    sns.set(style="whitegrid")
    sns.set(font_scale=1.6, style="ticks")

    for stimuli_dir, reps_dir, figs_dir, stimuli_postfix, title \
            in zip(STIMULI_DIRS, REPS_DIRS, FIGS_DIRS,
                   STIMULI_POSTFIXES, TITLES):
        # Filename to load/save results database from/to:
        results_filename = RESULTS_DIR + 'pca' + stimuli_postfix + '.csv'
        correlations_results_filename = RESULTS_DIR + \
            'correlations_pca' + stimuli_postfix + '.csv'

        try:
            df = pd.read_csv(correlations_results_filename)
        except FileNotFoundError:
            print('Please run pca_permutations.py to have all files required'
                  'to create the figures.')
            continue

        # df.to_csv(correlations_results_filename)

        sns.set_palette(sns.color_palette('muted'))
        colors = ["bright sky blue", "purple"]
        sns.set_palette(sns.xkcd_palette(colors))

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.15)

        ax = df.plot.line(x='Layer', y=['Frequency',
                                        'Orientation'],
                          ax=ax, lw=3,  alpha=0.75,
                          legend=False)
        # ax.set_ylim(0.45,1)
        # plt.legend(loc=3)
        # y_min = 0.39
        # y_max = 1.005
        # plt.axis([-1, 26, y_min, y_max])
        #
        # sns.despine(offset=10, trim=True)
        # ax.set_xlabel('Network Layer', size=20)
        # ax.set_ylabel('Correlation Coefficient', size=20)
        mean_permuted_frequency = []
        mean_permuted_orientation = []
        for l, layer_name in enumerate(LAYER_NAMES):
            try:
                del permutation_df
            except NameError:
                None
            permutations_filename = (RESULTS_DIR + 'permutations_pca/'
                                     + 'permuted_correlations'
                                     + stimuli_postfix + '_layer_' + str(l)
                                     + '.csv')

    #         print(permutations_filename)

            try:
                permutation_df = pd.read_csv(permutations_filename,
                                             index_col=0)
            except FileNotFoundError:
                print('File missing:', permutations_filename)
                print('Please run pca_permutations.py to have all files'
                      ' required to create the figures.')
                continue

            mean_permuted_frequency.append(
                permutation_df['Permuted Frequency'].mean())
            mean_permuted_orientation.append(
                permutation_df['Permuted Orientation'].mean())

    #         print(mean_permuted_frequency)
    #         print(mean_permuted_orientation)
        ax.plot(mean_permuted_orientation, color='#aaaaaa', lw=3,
                linestyle='--', label='Mean Permuted Orientation')
        ax.plot(mean_permuted_frequency, color='#000000', lw=3,
                linestyle=':', label='Mean Permuted Frequency')
        plt.legend(loc=3, frameon=False)
        y_min = 0.1
        y_max = 1.005
    #     plt.axis([-1, 26, y_min, y_max])
        sns.despine(offset=10, trim=True)
        ax.set_xlabel('Network Layer', size=20)
        ax.set_ylabel('Correlation Coefficient', size=20)

        general_figs_dir = os.sep.join(figs_dir.split(os.sep)[:-2])
        general_figs_dir = os.sep.join(
            [general_figs_dir, figs_dir.split(os.sep)[-2]])

        general_figs_dir = os.sep.join(figs_dir.split(os.sep)[:-2])
        general_figs_dir = os.sep.join(
            [general_figs_dir, figs_dir.split(os.sep)[-2]])

        ax.tick_params(axis='both', which='both', top='off', right='off')

        fig.savefig(general_figs_dir + '_correlation.pdf', bbox_inches='tight')
        fig.savefig(general_figs_dir + '_correlation.png', bbox_inches='tight')
        # if 'original' in figs_dir:
        # print(df[['Frequency', 'Orientation']])
        # print(df)

        if just_original:
            print(general_figs_dir + '_correlation.pdf')
            return general_figs_dir + '_correlation.pdf'


if __name__ == '__main__':
    create_and_save_figures(just_original=False)
