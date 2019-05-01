from scipy import stats
"""Graph accuracies at layer 0 using Luce choice.

This code graphs the layer 0 (pixel space) accuracies for all combinations of
train and test using both the Luce choice and the optimal decision boundary. It
produces a combined figure with all 3 train/test combinations on the same axes.
"""

# from __future__ import division, print_function
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pylab as plt
from matplotlib.lines import Line2D

from utils.misc import LAYER_NAMES

from cardiogram_experiment.misc import (EXP_DIR, is_normal,
                                        get_optimum_accuracy_boundary,
                                        prob_correct)


TRAIN_TYPES = ['colour', 'colour', 'grayscale']
TEST_TYPES = ['colour', 'grayscale', 'grayscale']
LAYER_INDEX = 0
optimum_boundary_models_file = (EXP_DIR
                                + './optimum_boundary_models/all_models.csv')
# sns.set(style="whitegrid")
# sns.set(font_scale=1.2, style="ticks")


def opt_correct(row):
    """
    Return a new column 'Optimum Correct'.

    The new column is based on 'Optimum Healthy' and 'Category'.
    """
    if row['Category'] == 'Normal':
        row['Optimum Correct'] = row['Optimum Healthy']
    elif row['Category'] == 'Abnormal':
        row['Optimum Correct'] = np.abs(1 - row['Optimum Healthy'])
    return row


def create_models():
    for train, test in zip(TRAIN_TYPES, TEST_TYPES):
        model_label = 'Train: ' + train + '; test: ' + test
        print(model_label)

        # Part of directory to save results:
        train = '_train_' + train
        test = '_test_' + test

        # Depending on the testing images, they have either no postfix or one that
        # denotes they are grayscale:
        if 'grayscale' in test:
            test_postfix = '_grayscale'
        elif 'gray' in test:
            test_postfix = '_gray'
        else:
            test_postfix = ''

        # Ditto:
        if 'grayscale' in train:
            train_postfix = '_grayscale'
        elif 'gray' in train:
            train_postfix = '_gray'
        else:
            train_postfix = ''

        model_dir = EXP_DIR + '/exemplar_model' + train + test
        results_dir = model_dir + '/results/'

        results_base_filename = results_dir + str(LAYER_INDEX).zfill(2) +\
            '_' + LAYER_NAMES[LAYER_INDEX].replace("/", "_")
        model_df = pd.read_csv(results_base_filename + '_hold_one_out.csv')

        # Start with a bit of tidying:
        model_df.rename(columns={'Unnamed: 0': 'Image Names',
                                 'Ratio of Healthy to Everything':
                                 'Probability Healthy'},
                        inplace=True)
        # Add a column for amount of damage:
        model_df['Amount of Damage'] = \
            [int(l[- 11 - len(test_postfix) - len('_test'):
                   - 9 - len(test_postfix) - len('_test')])
             for l in list(model_df['Image Names'])]

        # Sort abnormal items by probability healthy:
        model_df.sort_values(
            ['Amount of Damage', 'Probability Healthy'], ascending=[True, True])

        model_df = model_df.apply(prob_correct, axis=1)

        healthy_image_names = []
        abnormal_image_names = []
        for image_name in model_df['Image Names']:
            if is_normal(image_name):
                healthy_image_names.append(image_name)
            else:
                abnormal_image_names.append(image_name)
        model_df['Model'] = model_label

        # Calculate optimal boundary using Brad's method of pairs that are left
        # out:
        mean_optimum_accuracy, mean_optimum_boundary = \
            get_optimum_accuracy_boundary(model_df)

        model_df['Optimum Healthy'] = (
            model_df['Probability Healthy'] > mean_optimum_boundary).astype(int)

        model_df = model_df.apply(opt_correct, axis=1)

        try:
            models_df = models_df.append(model_df)  # noqa
        except NameError:
            models_df = model_df
    models_df.to_csv(optimum_boundary_models_file)
    return models_df


def create_and_save_figures():
    sns.set()
    sns.set(style="whitegrid")
    sns.set(font_scale=1.2, style="ticks")

    #
    # sns.set(style="whitegrid")
    # sns.set(font_scale=1.28, style="ticks")
    try:
        models_df = pd.read_csv(optimum_boundary_models_file)
    except FileNotFoundError:
        models_df = create_models()

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ci = 0
    scatter_kws = {"s": 25, "alpha": 0.7}
    line_kws = {"alpha": 0.7}

    y_variable = 'Probability Healthy'
    x_variable = 'Amount of Damage'

    ax1 = sns.regplot(x=x_variable, y=y_variable,
                      data=models_df[models_df['Model'] ==
                                     'Train: colour; test: colour'], ci=ci,
                      scatter_kws=scatter_kws, line_kws=line_kws,
                      ax=ax1, color='#c63d92',
                      marker='.')
    ax1 = sns.regplot(x=x_variable, y=y_variable,
                      data=models_df[models_df['Model'] ==
                                     'Train: grayscale; test: grayscale'], ci=ci,
                      scatter_kws=scatter_kws, line_kws=line_kws,
                      ax=ax1, color='#888888',
                      marker='.')
    ax1 = sns.regplot(x=x_variable, y=y_variable,
                      data=models_df[models_df['Model'] ==
                                     'Train: colour; test: grayscale'], ci=ci,
                      scatter_kws=scatter_kws, line_kws=line_kws,
                      ax=ax1,
                      color='#EEA9E1', marker='.')

    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off
    ax1.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
    )

    #
    # y_variable = 'Probability Correct'
    #
    # ax2 = sns.regplot(x=x_variable, y=y_variable,
    #                   data=models_df[models_df['Model'] ==
    #                                  'Train: colour; test: colour'], ci=ci,
    #                   scatter_kws=scatter_kws, line_kws=line_kws,
    #                   ax=ax2, color='#c63d92', marker='.')
    #
    # ax2 = sns.regplot(x=x_variable, y=y_variable,
    #                   data=models_df[models_df['Model'] ==
    #                                  'Train: colour; test: grayscale'], ci=ci,
    #                   scatter_kws=scatter_kws, line_kws=line_kws,
    #                   ax=ax2, color='#EEA9E1', marker='.')
    #
    # ax2 = sns.regplot(x=x_variable, y=y_variable,
    #                   data=models_df[models_df['Model'] ==
    #                                  'Train: grayscale; test: grayscale'], ci=ci,
    #                   scatter_kws=scatter_kws, line_kws=line_kws,
    #                   ax=ax2, color='#888888', marker='.')

    custom_lines = [Line2D([0], [0], color='#c63d92', lw=2),
                    Line2D([0], [0], color='#888888', lw=2),
                    Line2D([0], [0], color='#EEA9E1', lw=2)]

    models = ['Train: color; test: color',
              'Train: grayscale; test: grayscale',
              'Train: color; test: grayscale',
              ]
    ax1.legend(custom_lines, models, loc='upper right', frameon=False)

    ax1.set_xlim([-5, 55])
    # ax2.set_xlim([-5, 55])
    sns.despine(offset=5, trim=True)
    fig.tight_layout()

    figure_name = EXP_DIR + '/figures/layer_' + str(LAYER_INDEX) + '_OLS.pdf'
    fig.savefig(figure_name, bbox_inches='tight', pad_inches=0.05)

    figure_name = EXP_DIR + '/figures/layer_' + str(LAYER_INDEX) + '_OLS.pdf'
    fig.savefig(figure_name, bbox_inches='tight', pad_inches=0.05)

    # get coeffs of linear fit
    df = models_df[models_df['Model'] ==
                   'Train: colour; test: grayscale']
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['Amount of Damage'], df['Probability Healthy'])
    print(slope, intercept, r_value, p_value, std_err)

    return figure_name


if __name__ == '__main__':
    create_and_save_figures()
