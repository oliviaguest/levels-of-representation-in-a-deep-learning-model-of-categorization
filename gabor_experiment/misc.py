"""Default values and functions called from more than one file."""

import os
import glob

EXP_DIR = './gabor_experiment/'
RESULTS_DIR = EXP_DIR + 'results/'

ORIGINAL_STIMULI_DIR = EXP_DIR + 'stimuli/original/'
LEFT_STIMULI_DIR = EXP_DIR + 'stimuli/left/'
RIGHT_STIMULI_DIR = EXP_DIR + 'stimuli/right/'

ORIGINAL_REPS_DIR = EXP_DIR + 'layer_representations/original/'
LEFT_REPS_DIR = EXP_DIR + 'layer_representations/left/'
RIGHT_REPS_DIR = EXP_DIR + 'layer_representations/right/'

FIGURES_DIR = EXP_DIR + 'figures/'
ORIGINAL_FIGURES_DIR = FIGURES_DIR + 'original/'
LEFT_FIGURES_DIR = FIGURES_DIR + 'left/'
RIGHT_FIGURES_DIR = FIGURES_DIR + 'right/'

STIMULI_POSTFIXES = ['', '_left', '_right']

STIMULI_DIRS = [ORIGINAL_STIMULI_DIR, LEFT_STIMULI_DIR, RIGHT_STIMULI_DIR]
REPS_DIRS = [ORIGINAL_REPS_DIR, LEFT_REPS_DIR, RIGHT_REPS_DIR]
FIGS_DIRS = [ORIGINAL_FIGURES_DIR, LEFT_FIGURES_DIR, RIGHT_FIGURES_DIR]
TITLES = ['Original Stimuli', 'Left Stimuli', 'Right Stimuli']


ALPHA = 0.01
PERMUTATIONS = 10000


def get_subset(stimuli_dir, basename=False):
    """Return the subset of stimuli we are using, one hundredth of total."""
    # Grab all the stimuli names:
    stimuli = glob.glob(stimuli_dir + '*.jpg')
    stimuli.sort()
    # We are only using a subset of stimuli (a hundredth):
    subset = []
    for stimulus in stimuli:
        if int(os.path.basename(stimulus)[1:3]) % 10 == 0:
            # subset.append(stimulus)
            if (int(os.path.basename(stimulus)[4:6]) - 1) % 10 == 0:
                subset.append(stimulus)
    if basename:
        subset = [os.path.basename(os.path.splitext(s)[0]) for s in subset]
    return subset


def get_representations(stimuli, reps_dir):
    """Return the paths to the stimuli's representations on each layer."""
    from utils.misc import LAYER_NAMES
    subset_paths = [[] for l in LAYER_NAMES]
    for l, layer_name in enumerate(LAYER_NAMES):
        for stimulus in stimuli:
            subset_paths[l].append(reps_dir +
                                   layer_name.replace("/", "_") +
                                   '/' + stimulus + '.csv')
    return subset_paths


def max_correlation(row, column_name):
    max_value = row[['Correlation between ' + column_name + ' and 1st PC',
                     'Correlation between ' + column_name + ' and 2nd PC']]\
        .abs().argmax()
    return row[['Correlation between ' + column_name + ' and 1st PC',
                'Correlation between ' + column_name + ' and 2nd PC']]\
        .abs().max()
