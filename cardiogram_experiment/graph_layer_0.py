"""Graph accuracies at layer 0 using Luce choice and optimal decision boundary.

This code graphs the layer 0 (pixel space) accuracies for all combinations of
train and test using both the Luce choice and the optimal decision boundary. It
produces 2 types of figures: as a triptych (3 subfigures side-by-side) and as a
combined figure with all 3 train/test combinations on the same axes.
"""

from __future__ import division, print_function
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt

from utils.misc import LAYER_NAMES

from cardiogram_experiment.misc import (EXP_DIR, is_normal,
                                        get_optimum_accuracy_boundary,
                                        prob_correct)


TRAIN_TYPES = ['colour', 'colour', 'grayscale']
TEST_TYPES = ['colour', 'grayscale', 'grayscale']
LAYER_INDEX = 0
sns.set(style="whitegrid")
sns.set(font_scale=1.05, style="ticks")


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


def create_and_save_figures(models_df, save_dir, y_variable, loc=False,
                            ci=None, col_wrap=None, col=None):
    """"Create and save figures for accuracies at given layer."""
    p = sns.lmplot(x='Amount of Damage', y=y_variable, hue="Model",
                   data=models_df, ci=ci, palette="muted", legend=False,
                   scatter_kws={"s": 50, "alpha": 0.75}, col_wrap=col_wrap,
                   col=col)
    p.fig.tight_layout()
    if loc:
        plt.legend(loc=loc, framealpha=0.8, frameon=False,
                   fontsize=10)
    sns.despine(offset=10, trim=True)

    plt.savefig(save_dir + '.png', bbox_inches='tight')
    plt.savefig(save_dir + '.pdf', bbox_inches='tight')


partial_figure_name = EXP_DIR + '/figures/layer_' + str(LAYER_INDEX)
print(partial_figure_name)
# Show the results of a linear regression on each train/test combination as a
# triptych, with accuracy calculated using Luce choice:
create_and_save_figures(models_df,
                        (partial_figure_name + '_scatterplot_triptych_luce'),
                        'Probability Healthy',
                        ci=95,
                        col_wrap=3,
                        col='Model')

# Same as above but all lines/points on the same axes:
create_and_save_figures(models_df,
                        (partial_figure_name +
                         '_scatterplot_luce'),
                        'Probability Healthy',
                        ci=95,
                        loc='lower left')

# Same as above but regression on 'Probability Correct':
create_and_save_figures(models_df,
                        (partial_figure_name +
                         '_scatterplot_luce_correct'), 'Probability Correct',
                        ci=95, loc='lower left')

# Show the results of a linear regression on each train/test combination as a
# triptych, with accuracy calculated using the optimal decision boundary:
create_and_save_figures(models_df,
                        (partial_figure_name +
                         '_scatterplot_triptych_optimum'), 'Optimum Healthy',
                        ci=95, col_wrap=3, col='Model')

# Same as above but all lines/points on the same axes:
create_and_save_figures(models_df,
                        (partial_figure_name +
                         '_scatterplot_optimum'), 'Optimum Healthy', ci=95,
                        loc='lower left')

# Same as above but with regression on 'Optimum Correct':
create_and_save_figures(models_df,
                        (partial_figure_name +
                         '_scatterplot_optimum_correct'), 'Optimum Correct',
                        ci=95, loc='center left')
