"""Default values and functions called from more than one file."""
from __future__ import division, print_function

import random
import numpy as np

# Experiment directory:
EXP_DIR = './cardiogram_experiment/'

# Where to save the figures:
FIGURES_DIR = EXP_DIR + 'figures/'

# Where the stimuli/images are stored:
IMAGES_DIR = './cardiogram_experiment/stimuli/'

SUFFIXES = ['colour', 'gray', 'grayscale']


def get_stimuli_directories(g='color'):
    """Return the directories for the different stimuli."""
    if 'colo' in g:  # to allow for spelling variants
        return [
            'Pretraining', 'Set_A/Abnormal', 'Set_B/Abnormal', 'Set_A/Normal',
            'Set_B/Normal'
        ]
    elif g == 'grayscale':  # my creation
        grayscale = '_grayscale'
        return ['Pretraining' + grayscale, 'Set_A' + grayscale + '/Abnormal',
                'Set_B' + grayscale + '/Abnormal',
                'Set_A' + grayscale + '/Normal',
                'Set_B' + grayscale + '/Normal']
    elif g == 'gray':  # ones shown to pigeons
        grayscale = '_grayscale'
        return ['Grayscale_Stimuli']


def is_normal(image_name):
    """Return True if stimulus is normal/healthy."""
    if ('00' in image_name or '01' in image_name or
            '02' in image_name or '03' in image_name):
        return True
    return False


def prob_correct(row):
    """
    Return a new column: 'Probability Correct'.

    The new column is based on 'Probability Healthy' and 'Category'.
    """
    if row['Category'] == 'Normal':
        row['Probability Correct'] = row['Probability Healthy']
    elif row['Category'] == 'Abnormal':
        row['Probability Correct'] = np.abs(1 - row['Probability Healthy'])
    return row


def calculate_performance(df, boundary):
    """Return the true/false positives/negatives, given appropriate columns."""
    # Use the boundary to class the items:
    classed_as_healthy = df[df['Probability Healthy'] >
                            boundary][['Category',
                                       'Probability ' +
                                       'Healthy']]
    classed_as_unhealthy = df[df['Probability Healthy'] <=
                              boundary][['Category',
                                         'Probability ' +
                                         'Healthy']]
    healthy_counts = classed_as_healthy.groupby(
        'Category').count().transpose()
    unhealthy_counts = classed_as_unhealthy.groupby(
        'Category').count().transpose()
    try:
        true_positive = unhealthy_counts['Abnormal'].iloc[0]
    except KeyError:
        true_positive = 0
    try:
        true_negative = healthy_counts['Normal'].iloc[0]
    except KeyError:
        true_negative = 0
    try:
        false_positive = healthy_counts['Abnormal'].iloc[0]
    except KeyError:
        false_positive = 0
    try:
        false_negative = unhealthy_counts['Normal'].iloc[0]
    except KeyError:
        false_negative = 0
    return true_positive, true_negative, false_positive, false_negative


def get_normal_and_abnormal_names(model_df):
    """Return the normal and abnormal names of the images in the dataframe."""
    healthy_image_names = []
    abnormal_image_names = []
    for image_name in model_df['Image Names']:
        if is_normal(image_name):
            healthy_image_names.append(image_name)
        else:
            abnormal_image_names.append(image_name)
    return healthy_image_names, abnormal_image_names


def get_optimum_accuracy_boundary(model_df):
    """Calculate the boundary which gives the optimum accuracy."""
    accuracies_for_all_images = []
    boundaries_for_all_images = []

    # NOTE: this is hardcoded due to the nature of the image names!
    severity = np.asarray([l[-11 - len('_test'): -9 - len('_test')]
                           for l in model_df['Image Names']])
    model_df['Severity'] = severity
    model_df.sort_values(by='Severity', inplace=True)

    healthy_image_names, abnormal_image_names = \
        get_normal_and_abnormal_names(model_df)

    # For each item in both categories:
    for image in healthy_image_names + abnormal_image_names:
        # Assign an item randomly from the other category to make a pair which
        # are to be left out:
        if is_normal(image):
            abnormal = random.choice(abnormal_image_names)
            normal = image
        else:
            normal = random.choice(healthy_image_names)
            abnormal = image
        # Now create a new dataframe with all the rest of the items (pair
        # above removed):
        rest = ((model_df['Image Names'] != normal) &
                (model_df['Image Names'] != abnormal))
        rest_df = model_df[rest]

        accuracy_for_current_image = 0
        for i, _ in enumerate(rest_df.index):
            # Skip the first because it is comparing pairs of items:
            if i == 0:
                continue
            # Take the midpoint of the probability healthy of item i-1 and
            # i as the boundary:
            test_boundary = (
                rest_df['Probability Healthy'].iloc[i] +
                rest_df['Probability Healthy'].iloc[i - 1]) / 2

            # Get the performance of the model with the current boundary:
            true_positive, true_negative, false_positive, false_negative = \
                calculate_performance(rest_df, test_boundary)

            # print(true_positive, true_negative, false_positive,
            #       false_negative)
            # print(rest_df)
            if (true_positive + true_negative) / \
                (true_positive + true_negative +
                 false_positive + false_negative) > \
                    accuracy_for_current_image:
                # If we just discovered a better boundary, i.e., with higher
                # accuracy, then we update accuracy and boundary:
                accuracy_for_current_image = \
                    (true_positive + true_negative) / \
                    (true_positive + true_negative +
                     false_positive + false_negative)
                boundary_for_current_image = test_boundary
            # A test:
            assert (true_positive + true_negative +
                    false_positive + false_negative) == \
                len(healthy_image_names + abnormal_image_names) - 2

        accuracies_for_all_images.append(accuracy_for_current_image)

        try:
            boundaries_for_all_images.append(boundary_for_current_image)
        except UnboundLocalError:
            print('No boundary because accuracy is:',
                  (true_positive + true_negative) /
                  (true_positive + true_negative +
                   false_positive + false_negative))

    # print(accuracies_for_all_images, boundaries_for_all_images)
    return (np.array(accuracies_for_all_images).mean(),
            np.array(boundaries_for_all_images).mean())
