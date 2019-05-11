"""Do the following:

(1) get an N (each item) x number of units matrix.
(2) Do a PCA.
(3) Plot the amount of variance explained for different number of components
    included.
(4) create two vectors of N dimensions.
(a) one vector is the orientation value of the stimulus orientation (1-90).
(b) the other is width (1-90).
(5) take one of these vectors and correlate with one of the two first principal
    component for a 2x2 correlation matrix. E.g., if orientation vector
    correlations with second component, then that component represents
    orientation.
(6) do a permutation test on the correlations.
"""

from __future__ import division, print_function

import os.path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from gabor_experiment.misc import (get_subset,
                                   ORIGINAL_REPS_DIR,
                                   LEFT_REPS_DIR, RIGHT_REPS_DIR,
                                   LEFT_FIGURES_DIR, RIGHT_FIGURES_DIR,
                                   STIMULI_POSTFIXES, RESULTS_DIR, FIGURES_DIR,
                                   STIMULI_DIRS, REPS_DIRS, FIGS_DIRS, TITLES,
                                   ALPHA, PERMUTATIONS, max_correlation)

from utils.misc import LAYER_NAMES


sns.set(font_scale=1.05, style="ticks")
sns.set_palette("Set1", 6, .75)


def run_and_plot_pca(representations_path, save_filename):
    """Run and plot PCA and return transformed representations."""
    # Keep track of the representations:
    representations = []
    for path in representations_path:
        df = pd.read_csv(path)  # noqa
        representations.append(df.T.values[0])
    # Do the PCA thing:
    pca = PCA()
    transformed_representations = pca.fit_transform(np.array(representations))
    variance_explained = pca.explained_variance_ratio_
    # Figure:
    fig = plt.figure()  # noqa
    plt.plot(variance_explained, '.-')
    sns.despine(fig)
    plt.ylabel('Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylim([-0.01, 0.4])
    plt.xlim([-1, 82])
    fig.savefig(save_filename + '.pdf',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(save_filename + '.png',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return transformed_representations


def run_pca_and_permutation_test():
    """
    Run all the required analyses and create the figures.
    Permutations argument refers to the number of times to permute the labels
    in the analysis."""
    for stimuli_dir, reps_dir, figs_dir, stimuli_postfix, title \
            in zip(STIMULI_DIRS, REPS_DIRS, FIGS_DIRS,
                   STIMULI_POSTFIXES, TITLES):
        # Filename to load/save results database from/to:
        results_filename = RESULTS_DIR + 'pca' + stimuli_postfix + '.csv'
        correlations_results_filename = (RESULTS_DIR + 'correlations_pca'
                                         + stimuli_postfix + '.csv')

        try:
            print(results_filename)

            df = pd.read_csv(results_filename)
            try:
                df = df.drop(columns=['Unnamed: 0'], axis=1)
            except KeyError:
                None

            dtype = {}
            keys = df.columns

            # values =
            for i in keys:
                if i == 'Image':
                    dtype[i] = str
                elif i == 'Frequency' or i == 'Orientation':
                    dtype[i] = int
                else:
                    dtype[i] = float
            df = df.astype(dtype)

        except FileNotFoundError:

            # Get the labels for the stimuli we want to use:
            subset = get_subset(stimuli_dir, True)

            # Collect up the paths to these stimuli's representations on each
            # layer:
            subset_paths = [[] for l in LAYER_NAMES]
            for l, layer_name in enumerate(LAYER_NAMES):
                for stimulus in subset:
                    subset_paths[l].append(reps_dir
                                           + layer_name.replace("/", "_")
                                           + '/' + stimulus + '.csv')

            # Set up the dataframe:
            data = {'Image': subset,
                    'Frequency': [int(img[1:3]) for img in subset],
                    'Orientation': [int(img[4:6]) for img in subset]}
            df = pd.DataFrame.from_dict(data)

            for l, layer_name in enumerate(LAYER_NAMES):
                save_filename = figs_dir + 'pca_layer_' + str(l)
                print('Running PCA on layer: ' + l + layer_name)
                # Run PCA, save plot of variance explained per component, and
                # return transformed representations:
                transformed_x = run_and_plot_pca(subset_paths[l],
                                                 save_filename)
                df['1st PC Layer ' + str(l)] = transformed_x[:, 0]
                df['2nd PC Layer ' + str(l)] = transformed_x[:, 1]
            df.to_csv(results_filename, index=False)

        # Set up the dataframe for the true correlations:
        data = {'Layer': LAYER_NAMES,
                'Correlation between Frequency and 1st PC':
                    [df.Frequency.corr(df['1st PC Layer ' + str(l)])
                     for l in range(len(LAYER_NAMES))],
                'Correlation between Frequency and 2nd PC':
                    [df.Frequency.corr(df['2nd PC Layer ' + str(l)])
                     for l in range(len(LAYER_NAMES))],
                'Correlation between Orientation and 1st PC':
                    [df.Orientation.corr(df['1st PC Layer ' + str(l)])
                        for l in range(len(LAYER_NAMES))],
                'Correlation between Orientation and 2nd PC':
                    [df.Orientation.corr(df['2nd PC Layer ' + str(l)])
                        for l in range(len(LAYER_NAMES))],

                'P-value for Correlation between Frequency and 1st PC':
                    [pearsonr(df.Frequency, df['1st PC Layer ' + str(l)])[1]
                     for l in range(len(LAYER_NAMES))],
                'P-value for Correlation between Frequency and 2nd PC':
                    [pearsonr(df.Frequency, df['2nd PC Layer ' + str(l)])[1]
                     for l in range(len(LAYER_NAMES))],
                'P-value for Correlation between Orientation and 1st PC':
                    [pearsonr(df.Orientation, df['1st PC Layer ' + str(l)])[1]
                        for l in range(len(LAYER_NAMES))],
                'P-value for Correlation between Orientation and 2nd PC':
                    [pearsonr(df.Orientation, df['2nd PC Layer ' + str(l)])[1]
                        for l in range(len(LAYER_NAMES))],

                }

        correlation_df = pd.DataFrame.from_dict(data)

        correlation_df['Layer'] = correlation_df.index
        # Save the max correlation involving a PC and Frequency
        correlation_df['Frequency'] = \
            correlation_df.apply(
                max_correlation, column_name='Frequency', axis=1)
        # Save the max correlation involving a PC and Orientation
        correlation_df['Orientation'] = \
            correlation_df.apply(max_correlation,
                                 column_name='Orientation', axis=1)
        correlation_df.to_csv(correlations_results_filename, index=False)

        # Now we create the permutations per layer:
        for l, layer_name in enumerate(LAYER_NAMES):

            permutations_filename = (RESULTS_DIR + 'permutations_pca/'
                                     + 'permuted_correlations' + stimuli_postfix
                                     + '_layer_' + str(l) + '.csv')
            try:
                permutation_df = pd.read_csv(permutations_filename)
            except FileNotFoundError:

                for permutation in range(PERMUTATIONS):
                    print(permutation)
                    shuffled_image_labels = df['Image']
                    shuffled_image_labels = shuffled_image_labels.sample(
                        frac=1)
                    shuffled_image_labels.reset_index(inplace=True, drop=True)

                    df['Permuted Image'] = shuffled_image_labels
                    df['Permuted Frequency'] = [int(img[1:3])
                                                for img in
                                                df['Permuted Image']]
                    df['Permuted Orientation'] = [int(img[4:6])
                                                  for img in
                                                  df['Permuted Image']]

                    # Now we set up the dataframe for the permutations:
                    columns = {
                        'Correlation between Permuted Frequency and 1st PC':
                        [df['Permuted Frequency'].corr(
                            df['1st PC Layer ' + str(l)])],
                        'Correlation between Permuted Frequency and 2nd PC':
                        [df['Permuted Frequency'].corr(
                            df['2nd PC Layer ' + str(l)])],
                        'Correlation between Permuted Orientation and 1st PC':
                        [df['Permuted Orientation'].corr(
                            df['1st PC Layer ' + str(l)])],
                        'Correlation between Permuted Orientation and 2nd PC':
                        [df['Permuted Orientation'].corr(
                            df['2nd PC Layer ' + str(l)])],

                        'P-value for Correlation between Permuted Frequency and 1st PC':
                        [pearsonr(df['Permuted Frequency'],
                                  df['1st PC Layer ' + str(l)])[1]],
                        'P-value for Correlation between Permuted Frequency and 2nd PC':
                        [pearsonr(df['Permuted Frequency'],
                                  df['2nd PC Layer ' + str(l)])[1]],
                        'P-value for Correlation between Permuted Orientation and 1st PC':
                        [pearsonr(df['Permuted Orientation'],
                                  df['1st PC Layer ' + str(l)])[1]],

                        'P-value for Correlation between Permuted Orientation and 2nd PC':
                        [pearsonr(df['Permuted Orientation'],
                                  df['2nd PC Layer ' + str(l)])[1]],


                    }
                    try:
                        permutation_df = permutation_df.append(
                            pd.DataFrame(columns, index=[permutation]))
                    except NameError:
                        permutation_df = pd.DataFrame(
                            columns, index=[permutation])

                # As before select the max Frequency and Orientation:
                permutation_df['Permuted Frequency'] = \
                    permutation_df.apply(max_correlation,
                                         column_name='Permuted Frequency', axis=1)
                permutation_df['Permuted Orientation'] = \
                    permutation_df.apply(max_correlation,
                                         column_name='Permuted Orientation', axis=1)
                permutation_df.to_csv(permutations_filename)
                del permutation_df

        for l, layer_name in enumerate(LAYER_NAMES):
            try:
                del permutation_df
            except NameError:
                None
            permutations_filename = (RESULTS_DIR + '/permutations_pca/' +
                                     'permuted_correlations' +
                                     stimuli_postfix + '_layer_' + str(l) +
                                     '.csv')
            permutations_figure_filename = (FIGURES_DIR
                                            + stimuli_postfix[1:]
                                            + '/permuted_correlations'
                                            + stimuli_postfix + '_layer_'
                                            + str(l) + '.pdf')
            permutation_df = pd.read_csv(
                permutations_filename, index_col=0)

            fig, ax = plt.subplots()
            for permutation_column in permutation_df.columns:
                sns.distplot(
                    permutation_df[permutation_column], ax=ax, label=permutation_column)
                ax.set_title(layer_name)
            ax.set_xlabel('Correlation')
            ax.legend()
            fig.savefig(permutations_figure_filename)
            # plt.show()
            plt.close()

        # Now we want to calculate the permuted p-values for the two values we
        # picked for Frequency and Orientation.
        for column in ['Frequency', 'Orientation']:
            # print(column)
            new_column = 'P-value for ' + \
                column + ' Less Than ' + str(ALPHA)
            permuted_p_value_less_than_alpha = []
            permuted_column = 'Permuted ' + column
            for l, layer_name in enumerate(LAYER_NAMES):
                try:
                    del permutation_df
                except NameError:
                    None
                permutations_filename = (RESULTS_DIR + '/permutations_pca/' +
                                         'permuted_correlations' +
                                         stimuli_postfix + '_layer_' + str(l) +
                                         '.csv')
                permutations_figure_filename = (FIGURES_DIR
                                                + stimuli_postfix[1:]
                                                + '/permuted_correlations'
                                                + stimuli_postfix + '_layer_'
                                                + str(l) + '.pdf')
                permutation_df = pd.read_csv(permutations_filename)

                permutation_df = permutation_df.sort_values(
                    by=[permuted_column])
                permutation_df = permutation_df.reset_index()
                del permutation_df['index']
                high_alpha_cut_off = int(
                    len(permutation_df[permuted_column]) - ALPHA
                    * len(permutation_df[permuted_column]))

                correlation = float(
                    correlation_df[correlation_df['Layer'] == l][column])
                high_permutation_correlation_cut_off = permutation_df[permuted_column]\
                    .iloc[high_alpha_cut_off]

                permuted_p_value_less_than_alpha.append(
                    high_permutation_correlation_cut_off < correlation)
            correlation_df[new_column] = permuted_p_value_less_than_alpha

        correlation_df['Layer'] = correlation_df.index

        correlation_df.to_csv(correlations_results_filename, index=False)


if __name__ == '__main__':
    run_pca_and_permutation_test()
