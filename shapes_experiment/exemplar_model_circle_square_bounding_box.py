"""Run exemplar model on stimuli with bounding boxes."""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import seaborn as sns

from utils.misc import LAYER_NAMES
from shapes_experiment.misc import (EXP_DIR, IMAGES_SUB_DIRS, ACCURACY_DIR,
                                    IMAGES_SUB_SUB_DIRS, get_dimensions_df,
                                    create_and_save_figures,
                                    get_optimum_correct)

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

def prototypes_are_opposite(p_as, p_bs):
    """Check the prototypes make sense given the experiment."""
    for a, b in zip(p_as, p_bs):
        if 'left' in a and 'left' in b:
            return False
        if 'right' in a and 'right' in b:
            return False
        if 'red' in a and 'red' in b:
            return False
        if 'blue' in a and 'blue' in b:
            return False
        if 'big' in a and 'big' in b:
            return False
        if 'small' in a and 'small' in b:
            return False
        if 'square' in a and 'square' in b:
            return False
        if 'circle' in a and 'circle' in b:
            return False
    return True

def categories_are_opposite(is_as, is_bs):
    """Check the categories make sense given the experiment."""
    for i_as, i_bs in zip(is_as, is_bs):
        # Check just one item and we can infer what everything should
        # be. If the first item in a is on the left, then all items in
        # a should be on the left and all items in b should be on the
        # right.
        if 'left' in i_as[0]:
            a_side = 'left'
            b_side = 'right'
        else:
            a_side = 'right'
            b_side = 'left'
        for a, b in zip(i_as, i_bs):
            if a_side in b:
                return False
            if b_side in a:
                return False
    return True


def create_all_categories(items):
    """Create all permutations of members & prototypes per category (A & B)."""
    items_as = []
    items_bs = []
    prototype_as = []
    prototype_bs = []

    # Create a list of the features each item has removing those which are
    # irrelevant:
    features_per_item = [item.split('_') for item in items]

    for i, item in enumerate(items):
        prototype_a = items[i]
        # If we have already done this combination but with B, go to the next
        # item:
        if prototype_a in prototype_bs:
            continue

        # If this item is not a prototype, skip it:
        if 'prototype' not in prototype_a:
            continue

        # Get the list of features:
        prototype_a_features = prototype_a.split('_')

        items_a = []
        items_b = []

        for features in features_per_item:
            # We do not want to add the prototypes to the items:
            if 'prototype' in features:
                continue

            # If the current item shares more than one feature with the
            # prototype for A:
            if len(set(prototype_a_features).intersection(set(features))) > 1:
                items_a.append(features)
            # Otherwise it shares one or fewer features, meaning it's in B:
            else:
                items_b.append(features)
            # If the current item shares no features with A then it is the
            # prototype for B by definition:
            if set(prototype_a_features).intersection(set(features)) ==\
                    set([]):
                prototype_b = '_'.join(features)
                if 'left' in prototype_a:
                    prototype_b = 'prototype_' + prototype_b + '_right'
                elif 'right' in prototype_a:
                    prototype_b = 'prototype_' + prototype_b + '_left'

        items_a = ['_'.join(item) for item in items_a]
        items_b = ['_'.join(item) for item in items_b]
        prototype_as.append(prototype_a)
        prototype_bs.append(prototype_b)
        items_as.append(items_a)
        items_bs.append(items_b)
    return prototype_as, items_as, prototype_bs, items_bs


if __name__ == '__main__':

    STIM_DIRS = [IMAGES_SUB_DIRS[1] + s for s in IMAGES_SUB_SUB_DIRS]
    # For the four different types of stimuli:
    for stim_dir in STIM_DIRS:
        stim_type = stim_dir.split('/')[0]
        hue = stim_dir.split('/')[1]
        # The following are the accuracies we care about:
        accuracy = []
        max_accuracy = []
        hue_accuracy = []
        size_accuracy = []
        shape_accuracy = []
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
        print(hue, stim_type)


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

        for layer_index, layer_name in enumerate(LAYER_NAMES):
            print('Layer:', layer_index, layer_name)

            results_base_filename = results_dir + str(layer_index).zfill(2) +\
                '_' + layer_name.replace("/", "_")
            figures_base_filename = figures_dir + str(layer_index).zfill(2) +\
                '_' + layer_name.replace("/", "_")

            # Load the representations for all inputs for a single layer:
            layer_filename = (EXP_DIR + 'layer_representations/' +
                              stim_dir + layer_name.replace("/", "_") +
                              '.csv')
            print('\tOpening CSV file with layer representations...')
            print(layer_filename)
            df = pd.read_csv(layer_filename)
            print('\tDone!')

            # Get the columns for items to create a train and test dataframe...
            items = list(df.columns)
            items.sort()
            stim_df = df[items]
            del df

            # Create all combinations of categories:
            prototype_as, items_as, prototype_bs, items_bs = \
                create_all_categories(
                    stim_df.columns)

            # For every possible prototype and category combination...
            for prototype_a, items_a, prototype_b, items_b in \
                    zip(prototype_as, items_as, prototype_bs, items_bs):

                # Create training and testing sets:
                test_df = stim_df[items_a + items_b]
                train_df = stim_df[[prototype_a, prototype_b]]

                # Calculate the similarity as correlation + 1:
                similarity = pd.concat([test_df, train_df], axis=1,
                                       join='inner')
                similarity = similarity.corr() + 1

                diagonal = similarity\
                    .values[([i for i in
                              range(len(similarity.columns))], [
                                  i for i in range(len(similarity))])]

                A_items_to_A_prototype = pd.DataFrame(
                    similarity[items_a].loc[prototype_a])
                A_items_to_B_prototype = pd.DataFrame(
                    similarity[items_a].loc[prototype_b])
                B_items_to_B_prototype = pd.DataFrame(
                    similarity[items_b].loc[prototype_b])
                B_items_to_A_prototype = pd.DataFrame(
                    similarity[items_b].loc[prototype_a])
                similarity_to_A_prototype = A_items_to_A_prototype.append(
                    B_items_to_A_prototype)
                similarity_to_B_prototype = A_items_to_B_prototype.append(
                    B_items_to_B_prototype)
                similarity_to_both = \
                    pd.DataFrame(similarity_to_A_prototype[prototype_a]
                                 + similarity_to_B_prototype[prototype_b])

                # Probability that an item is in A
                prob_A = pd.DataFrame(
                    similarity_to_A_prototype[prototype_a] /
                    similarity_to_both[0], columns=['Probability A'])

                # Probability that an item is in B
                prob_B = pd.DataFrame(
                    similarity_to_B_prototype[prototype_b] /
                    similarity_to_both[0], columns=['Probability B'])

                exemplar_model = pd.concat(
                    [prob_A, prob_B], axis=1, join='inner')

                exemplar_model = pd.concat([exemplar_model,
                                            similarity_to_A_prototype,
                                            similarity_to_B_prototype],
                                           axis=1, join='inner')

                exemplar_model \
                    .rename(columns={prototype_a:
                                     'Similarity to Prototype A',
                                     prototype_b:
                                     'Similarity to Prototype B'},
                            inplace=True)
                category = []
                for i in exemplar_model.index:
                    if i in items_a:
                        category.append('A')
                    elif i in items_b:
                        category.append('B')
                    else:
                        raise ValueError

                exemplar_model['Category'] = category

                # Dummy:
                exemplar_model['Probability Correct'] = \
                    exemplar_model['Category']

                exemplar_model.loc[items_a, 'Probability Correct'] = \
                    exemplar_model['Probability A'].loc[items_a]
                exemplar_model.loc[items_b, 'Probability Correct'] = \
                    exemplar_model['Probability B'].loc[items_b]
                accuracy.append(exemplar_model['Probability Correct'].mean())

                # Dummy:
                exemplar_model['Max Correct'] = exemplar_model['Category']

                exemplar_model.loc[items_a, 'Max Correct'] = \
                    exemplar_model['Probability A'].loc[items_a] > \
                    exemplar_model['Probability B'].loc[items_a]
                exemplar_model.loc[items_b, 'Max Correct'] = \
                    exemplar_model['Probability B'].loc[items_b] > \
                    exemplar_model['Probability A'].loc[items_b]
                max_accuracy.append(np.sum(exemplar_model['Max Correct']) /
                                    exemplar_model['Max Correct'].count())

                print(prototype_a)
                exemplar_model = get_optimum_correct(exemplar_model, items_a, items_b)
                print(exemplar_model)


                create_and_save_figures(exemplar_model, figures_base_filename,
                                        layer_index, layer_name, show=False)

                dim_df = get_dimensions_df(items_a, items_b, prototype_a,
                                           prototype_b)

                exemplar_model = pd.concat([exemplar_model, dim_df], axis=1,
                                           join='inner')

                hue_accuracy.append(exemplar_model[exemplar_model['Hue']
                                                   != exemplar_model[
                                                       'Prototype Hue']]
                                    [['Probability Correct']].mean())
                print(exemplar_model['Hue'])
                print(exemplar_model['Prototype Hue'])

                shape_accuracy.append(exemplar_model[exemplar_model['Shape']
                                                     != exemplar_model
                                                     ['Prototype Shape']][[
                                                         'Probability'
                                                         ' Correct']].mean())

                size_accuracy.append(exemplar_model[exemplar_model['Size']
                                                    != exemplar_model
                                                    ['Prototype Size']][[
                                                        'Probability'
                                                        ' Correct']].mean())

                same_hue_accuracy.append(exemplar_model[exemplar_model
                                                        ['Hue']
                                                        == exemplar_model
                                                        ['Prototype Hue']]
                                         [['Probability'
                                           ' Correct']].mean())

                same_shape_accuracy.append(exemplar_model[exemplar_model
                                                          ['Shape']
                                                          == exemplar_model
                                                          ['Prototype'
                                                           ' Shape']]
                                           [['Probability Correct']].mean())

                same_size_accuracy.append(exemplar_model[exemplar_model
                                                         ['Size']
                                                         == exemplar_model
                                                         ['Prototype'
                                                          ' Size']][[
                                                              'Probability'
                                                              ' Correct']]
                                          .mean())


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
                accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Max Accuracy', np.mean(max_accuracy[layer_index * (
                len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Optimum Accuracy', np.mean(
                optimum_accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Same Hue Luce Accuracy', np.mean(
                same_hue_accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Same Size Luce Accuracy', np.mean(
                same_size_accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Same Shape Luce Accuracy', np.mean(
                same_shape_accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Same Hue Optimum Accuracy', np.mean(
                optimum_same_hue_accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Same Size Optimum Accuracy', np.mean(
                optimum_same_shape_accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))
            accuracy_df.set_value(layer_name, 'Same Shape Optimum Accuracy', np.mean(
                optimum_same_size_accuracy[layer_index * (len(items_a) + len(items_b)):(layer_index + 1) * (len(items_a) + len(items_b))]))


        accuracy = np.asarray(accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)
        max_accuracy = np.asarray(max_accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)
        hue_accuracy = np.asarray(hue_accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)
        size_accuracy = np.asarray(size_accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)
        shape_accuracy = np.asarray(shape_accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)
        same_hue_accuracy = np.asarray(same_hue_accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)
        same_size_accuracy = np.asarray(same_size_accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)
        same_shape_accuracy = np.asarray(same_shape_accuracy)\
            .reshape((len(LAYER_NAMES),
                      len(items_a) + len(items_b))).mean(axis=1)

        accuracy_df.to_csv(ACCURACY_DIR + stim_dir.replace("/", "_") +
                           'accuracy.csv')
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'luce.csv', accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'max.csv', max_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'hue_luce.csv', hue_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'shape_luce.csv', shape_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'size_luce.csv', size_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'same_hue_luce.csv', same_hue_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'same_shape_luce.csv', same_shape_accuracy)
        np.savetxt(ACCURACY_DIR + stim_dir.replace("/", "_") +
                   'same_size_luce.csv', same_size_accuracy)
