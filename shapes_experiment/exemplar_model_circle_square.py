"""Run exemplar model on stimuli."""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import seaborn as sns

from utils.misc import LAYER_NAMES
from shapes_experiment.misc import (EXP_DIR, IMAGES_SUB_DIRS, ACCURACY_DIR,
                                    IMAGES_SUB_SUB_DIRS, get_dimensions_df,
                                    create_and_save_figures, get_optimum_correct)

import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None,
                        line=None):
    # https://stackoverflow.com/a/22376126
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(
        message, category, filename, lineno, line))
    exit()


warnings.showwarning = warn_with_traceback


sns.set(font_scale=1.8)
sns.set_style('ticks')


def create_all_categories(items):
    """Create all permutations of members & prototypes per category (A & B)."""
    categories_a = []
    categories_b = []
    prototypes_a = []
    prototypes_b = []
    for i, item in enumerate(items):
        prototype_a = items[i]
        if prototype_a in prototypes_b:
            continue
        prototype_a_features = prototype_a.split('_')
        features_per_item = [item.split('_') for item in items]

        category_a = []
        category_b = []
        for features in features_per_item:
            if len(set(prototype_a_features).intersection(set(features))) > 1:
                category_a.append(features)
            else:
                category_b.append(features)

            if set(prototype_a_features).intersection(set(features)) == \
                    set([]):
                prototype_b = '_'.join(features)

        category_a = ['_'.join(item) for item in category_a]
        category_b = ['_'.join(item) for item in category_b]
        prototypes_a.append(prototype_a)
        prototypes_b.append(prototype_b)
        categories_a.append(category_a)
        categories_b.append(category_b)
    return prototypes_a, categories_a, prototypes_b, categories_b


def create_and_save_exemplar_model(
        similarity_to_a, similarity_to_everything, results_base_filename,
        category_a, category_b):
    """Create and save an exemplar model."""
    exemplar_model = pd.DataFrame(
        {'Similarity to A': similarity_to_a,
         'Similarity to Everything': similarity_to_everything})
    exemplar_model['Ratio of A to Everything'] = \
        exemplar_model['Similarity to A'].divide(
            exemplar_model['Similarity to Everything'])
    category = []
    for i in exemplar_model.index:
        if i in category_a:
            category.append('A')
        elif i in category_b:
            category.append('B')
        else:
            raise ValueError
    exemplar_model['Category'] = category
    mean_exemplar_model = exemplar_model.groupby(
        ['Category'], as_index=False).mean()
    exemplar_model.to_csv(results_base_filename + '.csv')
    mean_exemplar_model.to_csv(
        results_base_filename + '_means.csv', index=False)


if __name__ == '__main__':
    STIM_DIRS = [IMAGES_SUB_DIRS[0] + s for s in IMAGES_SUB_SUB_DIRS]
    # For the four different types of stimuli:
    for stim_dir in STIM_DIRS:

        # Details of the stimuli to be used:
        stim_type = stim_dir.split('/')[0]
        hue = stim_dir.split('/')[1]


        # The following are the accuracies we care about:
        accuracy = []
        max_accuracy = []
        hue_accuracy = []
        shape_accuracy = []
        size_accuracy = []
        same_hue_accuracy = []
        same_size_accuracy = []
        same_shape_accuracy = []
        optimum_accuracy = []
        optimum_same_hue_accuracy = []
        optimum_same_size_accuracy = []
        optimum_same_shape_accuracy = []
        accuracy_df = pd.DataFrame(columns=['Hue Type',
                                            'Stimulus Type',
                                            'Luce Accuracy',
                                            'Max Accuracy',
                                            'Optimum Accuracy',
                                            'Same Hue Luce Accuracy',
                                            'Same Size Luce Accuracy',
                                            'Same Shape Luce Accuracy',
                                            'Same Hue Optimum Accuracy',
                                            'Same Size Optimum Accuracy',
                                            'Same Shape Optimum Accuracy'],
                                   index=LAYER_NAMES)
        accuracy_df['Stimulus Type'] = stim_type
        accuracy_df['Hue Type'] = hue

        # Create the required directories for saving:
        results_dir = EXP_DIR + '/results/' + stim_dir
        figures_dir = EXP_DIR + '/figures/' + stim_dir
        try:
            os.makedirs(figures_dir)
        except OSError:
            pass
        try:
            os.makedirs(results_dir)
        except OSError:
            pass

        # For each layer:
        for layer_index, layer_name in enumerate(LAYER_NAMES):
            print('Layer:', layer_index, layer_name)
            results_base_filename = results_dir + \
                str(layer_index).zfill(2) + '_' + layer_name.replace("/", "_")
            figures_base_filename = figures_dir + \
                str(layer_index).zfill(2) + '_' + layer_name.replace("/", "_")

            # Load the representations for all inputs for a single layer:
            layer_filename = EXP_DIR + 'layer_representations/' + \
                stim_dir + layer_name.replace("/", "_") + '.csv'
            df = pd.read_csv(layer_filename)

            # Create all category combinations:
            prototypes_a, categories_a, prototypes_b, categories_b = \
                create_all_categories(list(df.columns))
            # Loop through them:
            for prototype_a, category_a, prototype_b, category_b in zip(
                    prototypes_a, categories_a, prototypes_b, categories_b):

                # Make two lists of the items that aren't prototypes in A and
                # B:
                items_a = list(category_a)
                items_a.remove(prototype_a)
                items_b = list(category_b)
                items_b.remove(prototype_b)

                # Create training and testing sets:
                test_df = df[items_a + items_b]
                train_df = df[[prototype_a, prototype_b]]

                # Calculate the similarity as correlation + 1:
                similarity = pd.concat(
                    [test_df, train_df], axis=1, join='inner')
                similarity = similarity.corr() + 1

                # Calculate the similarities:
                a_items_to_a_prototype = pd.DataFrame(similarity[items_a].
                                                      loc[prototype_a])

                a_items_to_b_prototype = pd.DataFrame(similarity[items_a].
                                                      loc[prototype_b])

                b_items_to_b_prototype = pd.DataFrame(similarity[items_b].
                                                      loc[prototype_b])

                b_items_to_a_prototype = pd.DataFrame(similarity[items_b].
                                                      loc[prototype_a])

                similarity_to_a_prototype = a_items_to_a_prototype.append(
                    b_items_to_a_prototype)

                similarity_to_b_prototype = a_items_to_b_prototype.append(
                    b_items_to_b_prototype)

                similarity_to_both = \
                    pd.DataFrame(similarity_to_a_prototype[prototype_a]
                                 + similarity_to_b_prototype[prototype_b])
                prob_a = \
                    pd.DataFrame(
                        similarity_to_a_prototype[prototype_a] /
                        similarity_to_both[0], columns=['Probability A'])
                prob_b = pd.DataFrame(
                    similarity_to_b_prototype[prototype_b] /
                    similarity_to_both[0],
                    columns=['Probability B'])

                exemplar_model = pd.concat(
                    [prob_a, prob_b], axis=1, join='inner')

                exemplar_model = pd.concat([exemplar_model,
                                            similarity_to_a_prototype,
                                            similarity_to_b_prototype],
                                           axis=1, join='inner')
                exemplar_model.rename(
                    columns={prototype_a: 'Similarity to Prototype A',
                             prototype_b: 'Similarity to Prototype B'},
                    inplace=True)

                category = []
                for i in exemplar_model.index:
                    if i in category_a:
                        category.append('A')
                    elif i in category_b:
                        category.append('B')
                    else:
                        raise ValueError

                exemplar_model['Category'] = category

                exemplar_model['Probability Correct'] = \
                    exemplar_model['Category']  # dummy
                exemplar_model.loc[items_a, 'Probability Correct'] = \
                    exemplar_model['Probability A'].loc[items_a]
                exemplar_model.loc[items_b, 'Probability Correct'] = \
                    exemplar_model['Probability B'].loc[items_b]
                accuracy.append(exemplar_model['Probability Correct'].mean())

                exemplar_model['Max Correct'] = exemplar_model['Category']
                exemplar_model.loc[items_a, 'Max Correct'] = exemplar_model[
                    'Probability A'].loc[items_a] > \
                    exemplar_model['Probability B'].loc[items_a]
                exemplar_model.loc[items_b, 'Max Correct'] = exemplar_model[
                    'Probability B'].loc[items_b] > \
                    exemplar_model['Probability A'].loc[items_b]
                max_accuracy.append(
                    np.sum(exemplar_model['Max Correct']) /
                    exemplar_model['Max Correct'].count())
                print(prototype_a)
                exemplar_model = get_optimum_correct(exemplar_model, items_a, items_b)
                print(exemplar_model)

                # print('Luce choice accuracy at', layer_name,
                #       ':', accuracy[layer_index],)
                # print('Max accuracy at', layer_name,
                #       ':', max_accuracy[layer_index],)
                create_and_save_figures(exemplar_model, figures_base_filename,
                                        layer_index, layer_name, show = False)
                dim_df=get_dimensions_df(
                    items_a, items_b, prototype_a, prototype_b)
                exemplar_model = pd.concat(
                    [exemplar_model, dim_df], axis=1, join='inner')
                hue_accuracy \
                    .append(exemplar_model[exemplar_model['Hue'] !=
                                           exemplar_model['Prototype Hue']][[
                                               'Probability Correct']].mean())
                shape_accuracy \
                    .append(exemplar_model[exemplar_model['Shape'] !=
                                           exemplar_model['Prototype Shape']][[
                                               'Probability Correct']].mean())
                size_accuracy \
                    .append(exemplar_model[exemplar_model['Size'] !=
                                           exemplar_model['Prototype Size']][[
                                               'Probability Correct']].mean())
                same_hue_accuracy \
                    .append(exemplar_model[exemplar_model['Hue'] ==
                                           exemplar_model['Prototype Hue']][[
                                               'Probability Correct']].mean())
                same_shape_accuracy \
                    .append(exemplar_model[exemplar_model['Shape']
                                           == exemplar_model[
                                               'Prototype Shape']][[
                                                   'Probability Correct']]
                            .mean())
                same_size_accuracy \
                    .append(exemplar_model[exemplar_model['Size'] ==
                                           exemplar_model['Prototype Size']][[
                                               'Probability Correct']].mean())


                optimum_accuracy.append(np.sum(
                    exemplar_model['Optimum Correct']) / exemplar_model['Optimum Correct'].count())

                optimum_same_hue_accuracy.append(exemplar_model[exemplar_model
                                                                ['Hue']
                                                                == exemplar_model
                                                                ['Prototype Hue']]
                                                 [['Optimum'
                                                   ' Correct']].mean())

                optimum_same_shape_accuracy.append(exemplar_model[exemplar_model
                                                                  ['Shape']
                                                                  == exemplar_model
                                                                  ['Prototype'
                                                                      ' Shape']]
                                                   [['Optimum Correct']].mean())

                optimum_same_size_accuracy.append(exemplar_model[exemplar_model
                                                                 ['Size']
                                                                 == exemplar_model
                                                                 ['Prototype'
                                                                     ' Size']][[
                                                                         'Optimum'
                                                                         ' Correct']]
                                                  .mean())

            accuracy_df.set_value(layer_name, 'Luce Accuracy', np.mean(
                accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Max Accuracy', np.mean(max_accuracy[layer_index * (
                len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Optimum Accuracy', np.mean(
                optimum_accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Same Hue Luce Accuracy', np.mean(
                same_hue_accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Same Size Luce Accuracy', np.mean(
                same_size_accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Same Shape Luce Accuracy', np.mean(
                same_shape_accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Same Hue Optimum Accuracy', np.mean(
                optimum_same_hue_accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Same Size Optimum Accuracy', np.mean(
                optimum_same_shape_accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            accuracy_df.set_value(layer_name, 'Same Shape Optimum Accuracy', np.mean(
                optimum_same_size_accuracy[layer_index * (len(categories_a[0])):(layer_index + 1) * (len(categories_a[0]))]))
            print(accuracy_df)


        accuracy = np.asarray(accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)
        max_accuracy = np.asarray(max_accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)
        hue_accuracy = np.asarray(hue_accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)
        size_accuracy = np.asarray(size_accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)
        shape_accuracy = np.asarray(shape_accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)
        same_hue_accuracy = np.asarray(same_hue_accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)
        same_size_accuracy = np.asarray(same_size_accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)
        same_shape_accuracy = np.asarray(same_shape_accuracy) \
            .reshape((len(LAYER_NAMES), len(categories_a[0]))).mean(axis=1)

        accuracy_df.to_csv(ACCURACY_DIR + stim_dir.replace("/", "_") +
                           'accuracy.csv')
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'luce.csv', accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'max.csv', max_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'hue_luce.csv', hue_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'size_luce.csv', size_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'shape_luce.csv', shape_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'same_hue_luce.csv', same_hue_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'same_size_luce.csv', same_size_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'same_shape_luce.csv', same_shape_accuracy)
