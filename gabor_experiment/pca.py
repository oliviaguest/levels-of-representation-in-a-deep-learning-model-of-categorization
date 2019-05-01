"""Do the bulk of the following items.

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
"""

from gabor_experiment.misc import FIGS_DIRS
from gabor_experiment.misc import FIGURES_DIR
from gabor_experiment.misc import get_subset
from gabor_experiment.misc import REPS_DIRS
from gabor_experiment.misc import RESULTS_DIR
from gabor_experiment.misc import STIMULI_DIRS
from gabor_experiment.misc import STIMULI_POSTFIXES
from gabor_experiment.misc import TITLES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from utils.misc import LAYER_NAMES


sns.set(font_scale=1.05, style="ticks")


def run_and_plot_pca(representations_path, save_filename):
    """Run and plot PCA and return transformed representations."""
    # Keep track of the representations:
    representations = []
    for path in representations_path:
        df = pd.read_csv(path)
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


def main(shuffled=2):
    """Run all the required analyses and create the figures.

    The shuffled argument is the number of times to permute the labels in the
    analysis.
    """
    for stimuli_dir, reps_dir, figs_dir, stimuli_postfix, title \
            in zip(STIMULI_DIRS, REPS_DIRS, FIGS_DIRS,
                   STIMULI_POSTFIXES, TITLES):
        # Filename to load/save results database from/to:
        results_filename = RESULTS_DIR + 'pca' + stimuli_postfix + '.csv'
        correlations_results_filename = (RESULTS_DIR + 'correlations_pca' +
                                         stimuli_postfix + '.csv')

        try:
            df = pd.DataFrame.from_csv(results_filename)
        except IOError:

            # Get the labels for the stimuli we want to use:
            subset = get_subset(stimuli_dir, True)

            # Collect up the paths to these stimuli's representations on each
            # layer:
            subset_paths = [[] for l in LAYER_NAMES]
            for l, layer_name in enumerate(LAYER_NAMES):
                for stimulus in subset:
                    subset_paths[l].append(reps_dir +
                                           layer_name.replace("/", "_") +
                                           '/' + stimulus + '.csv')

            # Set up the dataframe:
            columns = [('Image', subset),
                       ('Frequency', [int(img[1:3]) for img in subset]),
                       ('Orientation', [int(img[4:6]) for img in subset])]
            df = pd.DataFrame.from_items(columns)
            for l, layer_name in enumerate(LAYER_NAMES):
                save_filename = figs_dir + 'pca_layer_' + str(l)
                print(l, layer_name, save_filename)
                # Run PCA, save plot of variance explained per component, and
                # return transformed representations:
                transformed_x = run_and_plot_pca(subset_paths[l],
                                                 save_filename)

                df['1st PC Layer ' + str(l)] = transformed_x[:, 0]
                df['2nd PC Layer ' + str(l)] = transformed_x[:, 1]

                print('Layer:', l, layer_name)
                print('Frequency, 1st PC Layer ' + str(l) + ':',
                      df.Frequency.corr(df['1st PC Layer ' + str(l)]))
                print('Orientation, 1st PC Layer ' + str(l) + ':',
                      df.Orientation.corr(df['1st PC Layer ' + str(l)]))

                print('Frequency, 2nd PC Layer ' + str(l) + ':',
                      df.Frequency.corr(df['2nd PC Layer ' + str(l)]))
                print('Orientation, 2nd PC Layer ' + str(l) + ':',
                      df.Orientation.corr(df['2nd PC Layer ' + str(l)]))
                print()
            df.to_csv(results_filename)

        for shuffle in range(shuffled + 1):
            print(shuffle)
            if shuffle:
                shuffle_text = 'Shuffled ' + str(shuffle) + ' '
                shuffled_image_labels = df['Image']
                shuffled_image_labels = shuffled_image_labels.sample(frac=1)
                shuffled_image_labels.reset_index(inplace=True, drop=True)
                df[shuffle_text + 'Image'] = shuffled_image_labels
                df[shuffle_text + 'Frequency'] = [int(img[1:3])
                                                  for img in
                                                  df[shuffle_text +
                                                      'Image']]
                df[shuffle_text + 'Orientation'] = [int(img[4:6])
                                                    for img in
                                                    df[shuffle_text +
                                                        'Image']]
            else:
                shuffle_text = ''

            # Add in the columns for the permutation test:

            # Set up the dataframe for the correlations:
            columns = [('Layer', LAYER_NAMES),
                       ('Correlation between ' + shuffle_text +
                        'Frequency and 1st PC',
                        [np.abs(df.Frequency.corr(df['1st PC Layer ' +
                                                     str(l)]))
                            for l in range(len(LAYER_NAMES))]),
                       ('Correlation between ' + shuffle_text +
                        'Frequency and 2nd PC',
                        [np.abs(df.Frequency.corr(df['2nd PC Layer ' +
                                                     str(l)]))
                         for l in range(len(LAYER_NAMES))]),
                       ('Correlation between ' + shuffle_text +
                        'Orientation and 1st PC',
                        [np.abs(df.Orientation.corr(df['1st PC Layer ' +
                                                       str(l)]))
                            for l in range(len(LAYER_NAMES))]),
                       ('Correlation between ' + shuffle_text +
                        'Orientation and 2nd PC',
                        [np.abs(df.Orientation.corr(df['2nd PC Layer '
                                                       + str(l)]))
                            for l in range(len(LAYER_NAMES))]),
                       ('P-value for Correlation between ' + shuffle_text +
                        'Frequency and 1st PC',
                        [pd.ols(y=df[shuffle_text + 'Frequency'],
                                x=df['1st PC Layer ' +
                                     str(l)], intercept=True).f_stat['p-value']
                         for l in range(len(LAYER_NAMES))]),
                       ('P-value for Correlation between ' + shuffle_text +
                        'Frequency and 2nd PC',
                        [pd.ols(y=df[shuffle_text + 'Frequency'],
                                x=df['2nd PC Layer ' +
                                     str(l)], intercept=True).f_stat['p-value']
                         for l in range(len(LAYER_NAMES))]),
                       ('P-value for Correlation between ' + shuffle_text +
                        'Orientation and 1st PC',
                        [pd.ols(y=df[shuffle_text + 'Orientation'],
                                x=df['1st PC Layer ' +
                                     str(l)], intercept=True).f_stat['p-value']
                         for l in range(len(LAYER_NAMES))]),
                       ('P-value for Correlation between ' + shuffle_text +
                        'Orientation and 2nd PC',
                        [pd.ols(y=df[shuffle_text + 'Orientation'],
                                x=df['2nd PC Layer ' +
                                     str(l)], intercept=True).f_stat['p-value']
                         for l in range(len(LAYER_NAMES))])]
            correlation_df = pd.DataFrame.from_items(columns)
            correlation_df['Layer'] = correlation_df.index
            correlation_df.to_csv(correlations_results_filename, index=False)

            # if not shuffled:
            if False:
                for column in correlation_df.columns:
                    if 'Correlation' in column and 'P-value' not in column:
                        figure_filename = (FIGURES_DIR +
                                           column.lower().replace(" ", "_")
                                           + stimuli_postfix)

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax = correlation_df.plot.scatter(x="Layer", y=column,
                                                         c='#4E78A0',
                                                         ax=ax)
                        ax = correlation_df.plot.scatter(x="Layer",
                                                         y='P-value for ' +
                                                         column,
                                                         c='r', marker='+',
                                                         ax=ax)
                        ax.fill([-1, 25.5, 25.5, -1],
                                [-.06, -.06, .05, .05], "r", alpha=0.05)
                        ax.fill([-1, 25.5, 25.5, -1],
                                [1.03, 1.03, .5, .5], c='#4E78A0', alpha=0.05)
                        plt.xlim([-1, 25.5])
                        plt.ylim([-.06, 1.03])
                        sns.despine(offset=10, trim=True)
                        plt.ylabel(column)
                        plt.title(title)
                        red = plt.scatter([], [], marker='+', color='r')
                        blue = plt.scatter([], [], marker='o', color='#4E78A0')
                        legend = plt.legend((blue, red), ('Correlation',
                                                          'P-value'),
                                            scatterpoints=1, frameon=True,
                                            bbox_to_anchor=(0.25, 0.3))
                        legend.get_frame().set_edgecolor('#000000')
                        legend.get_frame().set_alpha(0.3)
                        fig.savefig(figure_filename + '.pdf',
                                    bbox_inches='tight', pad_inches=0.05)
                        fig.savefig(figure_filename + '.png',
                                    bbox_inches='tight', pad_inches=0.05)
                        plt.close()


if __name__ == '__main__':
    main()
