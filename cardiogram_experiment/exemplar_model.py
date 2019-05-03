"""Run the exemplar models from the CLI."""

import argparse
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
from utils.misc import LAYER_NAMES
from cardiogram_experiment.misc import (EXP_DIR, is_normal, prob_correct,
                                        get_optimum_accuracy_boundary)

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set(font_scale=1.8)
sns.set_style('ticks')


def create_and_save_figures(exemplar_model, figures_base_filename, layer_index,
                            layer_name, show=False):
    """Create and save histogram for each layer."""
    # Make the histogram:
    fig, _ = plt.subplots(figsize=(8, 6))
    plt.hist(exemplar_model[exemplar_model['Category'] == 'Abnormal']
             ['Probability Healthy'].reset_index(drop=True), alpha=0.6,
             label="Abnormal")
    plt.hist(exemplar_model[exemplar_model['Category'] == 'Normal']
             ['Probability Healthy'].reset_index(drop=True), alpha=0.6,
             label="Normal")
    fig.suptitle('Layer: ' + str(layer_index) + ' ' + layer_name)
    plt.legend(loc='upper left', framealpha=0.8, frameon=True,
               fontsize=14)
    plt.xlabel('Probability Healthy')
    plt.ylabel('Counts')
    sns.despine()
    fig.tight_layout()
    plt.savefig(figures_base_filename + '_histogram.png', bbox_inches='tight')
    plt.savefig(figures_base_filename + '_histogram.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def run_and_graph_model(model_df, figures_base_filename, layer_index,
                        layer_name, postfix=''):
    """Run and graph the exemplar model."""
    # Start with a bit of tidying:
    model_df.rename(columns={'Unnamed: 0': 'Image Names',
                             'Ratio of Healthy to Everything':
                             'Probability Healthy'},
                    inplace=True)

    damage = []
    for l in list(model_df['Image Names']):
        try:
            damage.append(int(l.split('_')[1]))
        except ValueError:
            damage.append(int(l.split('_')[2]))
    # Add a column for amount of damage:
    model_df['Amount of Damage'] = damage

    print('\tCreating and saving figures for ' + layer_name)

    create_and_save_figures(model_df, figures_base_filename, layer_index,
                            layer_name)
    # Sort abnormal items by probability healthy:
    model_df.sort_values(
        ['Amount of Damage', 'Probability Healthy'], ascending=[True, True])

    # Calculate optimal boundary using Brad's method of pairs that are left
    # out:
    mean_optimum_accuracy, mean_optimum_boundary = \
        get_optimum_accuracy_boundary(model_df)

    print("\tOptimal accuracy: {0:.3g};"
          " optimal boundary: {1:.3g}".format(mean_optimum_accuracy,
                                              mean_optimum_boundary))

    # Calculate Luce choice accuracy:
    model_df = model_df.apply(prob_correct, axis=1)
    luce_accuracy = model_df['Probability Correct'].mean()
    print("\tLuce accuracy: {0:.3g}".format(luce_accuracy))

    return mean_optimum_accuracy, luce_accuracy


def create_and_save_exemplar_model(similarity_to_healthy,
                                   similarity_to_everything,
                                   results_base_filename):
    """Create and save the exemplar model."""
    exemplar_model = pd.DataFrame(
        {'Similarity to Healthy': similarity_to_healthy,
         'Similarity to Everything': similarity_to_everything})
    exemplar_model['Ratio of Healthy to Everything'] =\
        exemplar_model['Similarity to Healthy']\
        .divide(exemplar_model['Similarity to Everything'])
    is_healthy = []
    for i in exemplar_model.index:
        if is_normal(i):
            is_healthy.append('Normal')
        else:
            is_healthy.append('Abnormal')
    exemplar_model['Category'] = is_healthy
    mean_exemplar_model = exemplar_model.groupby(
        ['Category'], as_index=False).mean()
    exemplar_model.to_csv(results_base_filename + '.csv')
    mean_exemplar_model.to_csv(
        results_base_filename + '_means.csv', index=False)


if __name__ == '__main__':
    # This is to handle the passed CLI arguments:
    parser = argparse.ArgumentParser(
        description='Train and test an exemplar model.')
    parser.add_argument('train', metavar='train', type=str,
                        help='The name of the stimulus set to train on: '
                        'colour, gray, or grayscale.',
                        choices=['gray', 'grayscale', 'colour'])
    parser.add_argument('test', metavar='test', type=str,
                        help='The name of the stimulus set to test on: '
                        'colour, gray, or grayscale.',
                        choices=['gray', 'grayscale', 'colour'])
    args = parser.parse_args()

    # Set up variables for where to look for/save train/test sets:
    TRAIN = args.train
    TEST = args.test

    # Directory where I saved the output of the network:
    TRAIN_DIR = 'layer_representations_' + TRAIN
    TEST_DIR = 'layer_representations_' + TEST

    # Part of directory to save results:
    TRAIN = '_train_' + TRAIN
    TEST = '_test_' + TEST

    MODEL_DIR = EXP_DIR + '/exemplar_model' + TRAIN + TEST
    RESULTS_DIR = MODEL_DIR + '/results/'
    FIGURES_DIR = MODEL_DIR + '/figures/'

    # Make the directories, if they don't exist:
    try:
        os.makedirs(RESULTS_DIR)
    except OSError:
        pass
    try:
        os.makedirs(FIGURES_DIR)
    except OSError:
        pass

    # NOTE: There are two types of gray images because I initially made my own,
    # called "grayscale", from the colourised ones. Subsequently, Ed's lab sent
    # me the ones they actually used, called "gray". Luck had it that we
    # labelled them slightly differently so they can co-exist easily! The
    # results from these two are pretty much identical.

    # Depending on the testing images, they have either no postfix or one that
    # is "_grayscale"/"_gray":
    if 'grayscale' in TEST:
        TEST_POSTFIX = '_grayscale'
    elif 'gray' in TEST:
        TEST_POSTFIX = '_gray'
    else:
        TEST_POSTFIX = ''
    # Depending on the training images, they have either no postfix or one that
    # is "_grayscale"/"_gray":
    if 'grayscale' in TRAIN:
        TRAIN_POSTFIX = '_grayscale'
    elif 'gray' in TRAIN:
        TRAIN_POSTFIX = '_gray'
    else:
        TRAIN_POSTFIX = ''

    all_layers_optimal_accuracy = []
    all_layers_luce_accuracy = []

    for layer_index, layer_name in enumerate(LAYER_NAMES):
        print('Layer:', layer_index, layer_name)
        # File names for the results and figures:
        results_base_filename = RESULTS_DIR + str(layer_index).zfill(2) +\
            '_' + layer_name.replace("/", "_")
        figures_base_filename = FIGURES_DIR + str(layer_index).zfill(2) +\
            '_' + layer_name.replace("/", "_")

        try:
            # Open!
            model_df = pd.read_csv(
                results_base_filename + '_hold_one_out.csv')
        except IOError:
            # Not saved, so create!
            # Load the representations for all inputs for a single layer for
            # training:
            train_filename = (EXP_DIR + TRAIN_DIR + '/'
                              + layer_name.replace("/", "_") + '.csv')
            print('\tOpening CSV file for training...')
            try:
                train_df = pd.read_csv(train_filename)
            except FileNotFoundError:
                print('You need to run the deep network before this to get the'
                      ' layer representations for the stimuli!')
                exit()
            print('\tDone!')

            # And for testing:
            test_filename = (EXP_DIR + TEST_DIR + '/'
                             + layer_name.replace("/", "_") + '.csv')
            print('\tOpening CSV file for testing...')
            try:
                test_df = pd.read_csv(test_filename)
            except FileNotFoundError:
                print('You need to run the deep network before this to get the'
                      ' layer representations for the stimuli!')
                exit()
            print('\tDone!')

            # There is a list of numbers representing the index to the
            # dataframe, which I accidentally saved for some dataframes.
            # Useless to us, so drop:
            try:
                train_df = train_df.drop('Unnamed: 0', 1)
            except KeyError:
                pass
            try:
                test_df = test_df.drop('Unnamed: 0', 1)
            except KeyError:
                pass

            # For test set:
            # These are the names of the images each representation in this
            # layer belong to:
            test_image_labels = np.asarray(list(test_df))
            # Columns are not ordered by how healthy are they are, so create a
            # severity array with which to order them:
            severity = np.asarray([l[-11 - len(TEST_POSTFIX):]
                                   for l in test_image_labels])
            # Sort everything by severity, which is coded in the file name,
            # i.e., also in image_labels:
            sorted_indices = np.argsort(severity)
            severity = severity[sorted_indices]
            test_image_labels = test_image_labels[sorted_indices]
            # Then sort the dataframe itself by severity:
            test_df = test_df[test_image_labels]
            # Add a postfix to know what role these stimuli play:
            new_test_image_labels = [label + '_test' for label in
                                     test_image_labels]
            test_df.rename(columns=dict(zip(test_image_labels,
                                            new_test_image_labels)),
                           inplace=True)
            test_image_labels = new_test_image_labels

            # Same as above, but for training set:
            # These are the names of the images each representation in this
            # layer belong to:
            train_image_labels = np.asarray(list(train_df))
            # Columns are not ordered by how healthy are they are, so create a
            # severity array with which to order them:
            severity = np.asarray([l[-11 - len(TRAIN_POSTFIX):]
                                   for l in train_image_labels])
            # Sort everything by severity, which is coded in the file name,
            # i.e., also in image_labels:
            sorted_indices = np.argsort(severity)
            severity = severity[sorted_indices]
            train_image_labels = train_image_labels[sorted_indices]
            # Then sort the dataframe itself by severity:
            train_df = train_df[train_image_labels]
            # Add a postfix to know what role these stimuli play:
            new_train_image_labels = [label + '_train' for label in
                                      train_image_labels]
            train_df.rename(columns=dict(zip(train_image_labels,
                                             new_train_image_labels)),
                            inplace=True)
            train_image_labels = new_train_image_labels

            # Now for the exemplar model itself!
            # Calculate the similarity as correlation + 1, but first we need to
            # correlate test set with training set, so merge them:
            similarity = pd.concat([test_df, train_df],
                                   axis=1, join='inner')
            # Add the one:
            similarity = similarity.corr() + 1
            # Drop the correlations that are irrelevant:
            similarity = similarity.drop(test_image_labels)
            similarity = similarity.drop(train_image_labels, axis=1)
            # We also need the diagonal. What might superficially appeal to be
            # self-similarity, is not actually self, but can be grayscale item
            # versus its colourised counterpart, meaning it is not always 2:
            diagonal = similarity.values[
                ([i for i in range(similarity.shape[0])],
                 [i for i in range(similarity.shape[1])])]
            # Collect up all the columns which are healthy and unhealthy, for
            # easy access later on:
            test_healthy_column_names = []
            test_abnormal_column_names = []
            for column_name in test_df.columns:
                if is_normal(column_name):
                    test_healthy_column_names.append(column_name)
                else:
                    test_abnormal_column_names.append(column_name)
            # Using a hold one out strategy, create the exemplar model:
            # 1. How similar each abnormal item is to each healthy item.
            hold_one_out_abnormal_to_healthy = \
                similarity[test_abnormal_column_names].iloc[:len(
                    test_healthy_column_names)].sum()
            # 2.a. How similar each abnormal item is to each abnormal item.
            hold_one_out_abnormal_to_abnormal = \
                similarity[test_abnormal_column_names]. \
                iloc[len(test_healthy_column_names):(len(
                    test_healthy_column_names) + len(
                        test_abnormal_column_names))].sum()
            # 2.b. But not to itself!
            hold_one_out_abnormal_to_abnormal -= diagonal[len(
                test_healthy_column_names):(len(test_healthy_column_names) +
                                            len(test_abnormal_column_names))]
            # 2.c. Correct sum for missing item (self):
            hold_one_out_abnormal_to_abnormal *= \
                len(test_abnormal_column_names) / \
                (len(test_abnormal_column_names) - 1)

            # 3.a. How similar each healthy item is to each healthy item.
            hold_one_out_healthy_to_healthy = \
                similarity[test_healthy_column_names].iloc[:len(
                    test_healthy_column_names)].sum()
            # 3.b. But not to itself!
            hold_one_out_healthy_to_healthy -= diagonal[:len(
                test_healthy_column_names)]
            # 3.c. Correct sum for missing item (self):
            hold_one_out_healthy_to_healthy *= len(
                test_healthy_column_names) / (len(test_healthy_column_names) -
                                              1)
            # 4. How similar each healthy item is to each abnormal item.
            hold_one_out_healthy_to_abnormal = \
                similarity[test_healthy_column_names].iloc[len(
                    test_healthy_column_names):len(
                        test_healthy_column_names) + len(
                            test_abnormal_column_names)].sum()
            # 5. How similar each item is to each healthy item.
            hold_one_out_similarity_to_healthy = \
                hold_one_out_healthy_to_healthy.append(
                    hold_one_out_abnormal_to_healthy)
            # 6. How similar each item is to each abnormal item.
            hold_one_out_similarity_to_abnormal = \
                hold_one_out_healthy_to_abnormal.append(
                    hold_one_out_abnormal_to_abnormal)
            # 7. How similar each item is to each other item (with itself held
            # out).
            hold_one_out_similarity_to_all = \
                hold_one_out_similarity_to_healthy + \
                hold_one_out_similarity_to_abnormal

            # Save the exemplar model for future use:
            create_and_save_exemplar_model(hold_one_out_similarity_to_healthy,
                                           hold_one_out_similarity_to_all,
                                           (results_base_filename +
                                            '_hold_one_out'))

            # Ugly way of making sure to have the correct column names (index
            # needs to be called 'Unnamed: 0' for the renaming to work in
            # run_and_graph_model):
            model_df = pd.read_csv(
                results_base_filename + '_hold_one_out.csv')

        # Since we have it all, we can now graph and run the model and get back
        # the different accuracies depending on how we define hits and misses:
        optimal_accuracy, luce_accuracy = \
            run_and_graph_model(model_df, figures_base_filename, layer_index,
                                layer_name, postfix=TEST_POSTFIX)
        # Keep track of these accuracies per layer:
        all_layers_optimal_accuracy.append(optimal_accuracy)
        all_layers_luce_accuracy.append(luce_accuracy)

    # Save them to a file so they can be easily graphed:
    np.savetxt(MODEL_DIR + '/optimum_accuracy' + TRAIN + TEST + '.csv',
               all_layers_optimal_accuracy)
    np.savetxt(MODEL_DIR + '/luce_accuracy' + TRAIN + TEST + '.csv',
               all_layers_luce_accuracy)
