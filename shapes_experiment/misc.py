"""Default values and functions called from more than one file."""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Experiment directory:
EXP_DIR = './shapes_experiment/'

# Where to save the figures:
FIGURES_DIR = EXP_DIR + 'figures/'

# Where the stimuli/images are stored:
IMAGES_DIR = EXP_DIR + 'stimuli/'
IMAGES_SUB_DIRS = ['circle_square/', 'circle_square_bounding_box/']
IMAGES_SUB_SUB_DIRS = ['colour/', 'grayscale/']

# Accuracy directory:
ACCURACY_DIR = EXP_DIR + 'accuracy/'


def create_and_save_figure(accuracy_colour, accuracy_grayscale, filename,
                           y_min=0.48, y_max=0.59, y_label='Accuracy'):
    """Create figure for accuracy results."""

    fig = plt.figure()
    plt.plot(accuracy_colour, label='Color', color='#c63d92', linewidth=3)
    plt.plot(accuracy_grayscale, label='Grayscale',
             color='#999999', linewidth=3, alpha=0.6)
    plt.axis([-1, 26, y_min, y_max])
    sns.despine(offset=10, trim=True)
    plt.legend(loc=2)
    plt.xlabel('Network Layer', size=20)
    plt.ylabel(y_label, size=20)

    plot = fig.add_subplot(111)
    plot.tick_params(axis='both', which='major', labelsize=16)
    plot.tick_params(axis='both', which='minor', labelsize=20)
    fig.savefig(FIGURES_DIR + filename,
                format='pdf', bbox_inches='tight')
    return fig


def create_and_save_figures(exemplar_model, figures_base_filename,
                            layer_index, layer_name, show=False):
    """Create and save figures, specifically the histograms for each layer."""
    # Make the histogram for each layer:
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.hist(exemplar_model[exemplar_model['Category'] == 'B']
             ['Probability A'].reset_index(drop=True), alpha=0.6, label="B")
    plt.hist(exemplar_model[exemplar_model['Category'] == 'A']
             ['Probability A'].reset_index(drop=True), alpha=0.6, label="A")
    fig.suptitle('Layer: ' + str(layer_index) + ' ' + layer_name)
    plt.legend(loc='upper left', framealpha=0.8, frameon=True,
               fontsize=14)
    plt.xlabel('Probability A')
    plt.ylabel('Counts')
    sns.despine()
    fig.tight_layout()
    plt.savefig(figures_base_filename + '_histogram.png', bbox_inches='tight')
    plt.savefig(figures_base_filename + '_histogram.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def get_dimensions_df(items_A, items_B, prototype_A, prototype_B):
    """Get on which dimesion items differ with respect to the prototype."""
    items = items_A + items_B

    df = pd.DataFrame(index=items, columns=['Hue', 'Shape', 'Size',
                                            'Prototype Hue', 'Prototype Shape',
                                            'Prototype Size'])

    features_per_item = [item.split('_') for item in items]

    for i, features in enumerate(features_per_item):
        for feature in features:
            if (feature == 'red' or feature == 'blue' or
                    feature == 'dark' or feature == 'light'):
                df['Hue'].iloc[i] = feature
            elif feature == 'circle' or feature == 'square':
                df['Shape'].iloc[i] = feature
            elif feature == 'big' or feature == 'small':
                df['Size'].iloc[i] = feature

    prototype_A_features = prototype_A.split('_')
    prototype_B_features = prototype_B.split('_')

    def get_prototype_features(df, features, items):
        for feature in features:
            if (feature == 'red' or feature == 'blue' or
                    feature == 'dark' or feature == 'light'):
                df['Prototype Hue'].loc[items] = feature
            elif feature == 'circle' or feature == 'square':
                df['Prototype Shape'].loc[items] = feature
            elif feature == 'big' or feature == 'small':
                df['Prototype Size'].loc[items] = feature
        return df
    df = get_prototype_features(df, prototype_A_features, items_A)
    df = get_prototype_features(df, prototype_B_features, items_B)
    # df['Prototype Hue'].loc[items_A] = prototype_A_features[0]
    # df['Prototype Hue'].loc[items_B] = prototype_B_features[0]
    # df['Prototype Shape'].loc[items_A] = prototype_A_features[1]
    # df['Prototype Shape'].loc[items_B] = prototype_B_features[1]
    # df['Prototype Size'].loc[items_A] = prototype_A_features[2]
    # df['Prototype Size'].loc[items_B] = prototype_B_features[2]
    return df


def get_optimum_correct(exemplar_model, items_a, items_b):
    # Calculate the optimum boundary:
    exemplar_model = exemplar_model.sort_values(by=['Probability A'])
    # exemplar_model = exemplar_model.reset_index()
    boundary = 0
    temp_boundary = 0
    opt_accuracy = 0
    index = 0
    for row in exemplar_model.itertuples():
        if index == exemplar_model.shape[0] - 1:
            break
        exemplar_model.loc[items_a, 'Temp Optimum Correct'] = \
            exemplar_model.loc[items_a, 'Probability A'] > \
            temp_boundary
        exemplar_model.loc[items_b, 'Temp Optimum Correct'] = \
            exemplar_model.loc[items_b, 'Probability A'] < \
            temp_boundary

        temp_acc = np.sum(
            exemplar_model['Temp Optimum Correct']) / exemplar_model['Temp Optimum Correct'].count()
        if opt_accuracy < temp_acc:
            opt_accuracy = temp_acc
            boundary = temp_boundary
            exemplar_model['Optimum Correct'] = exemplar_model['Temp Optimum Correct']
        temp_boundary = (exemplar_model['Probability A'][index] +
                         exemplar_model['Probability A'][index + 1]) / 2
        index += 1

    # Now we have the optimum boundary:
    del exemplar_model['Temp Optimum Correct']

    exemplar_model.loc[items_a, 'Optimum Correct'] = \
        exemplar_model.loc[items_a, 'Probability A'] > \
        boundary
    exemplar_model.loc[items_b, 'Optimum Correct'] = \
        exemplar_model.loc[items_b, 'Probability A'] < \
        boundary

    return exemplar_model
