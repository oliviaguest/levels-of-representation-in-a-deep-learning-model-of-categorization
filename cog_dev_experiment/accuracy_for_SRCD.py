"""Calculate and graph the shape bias of the network at each layer."""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from utils.misc import LAYER_NAMES



def create_and_save_figures():
    EXP_DIR = './cog_dev_experiment/'

    sns.set(style="whitegrid")
    sns.set(font_scale=1.6, style="ticks")
    # Attempt to load the saved accuracies in order to graph them:
    try:
        luce_accuracy = np.loadtxt(EXP_DIR + 'luce_accuracy.csv')  # noqa
        max_accuracy = np.loadtxt(EXP_DIR + 'max_accuracy.csv')  # noqa
    # But if the files don't exist obviously we have to create and save them:
    except IOError:
        luce_accuracy = []  # noqa
        max_accuracy = []  # noqa
        for layer_index, layer_name in enumerate(LAYER_NAMES):
            # Load the representations for all inputs for a single layer:
            print('Layer:', layer_index, layer_name)
            layer_filename = EXP_DIR + 'layer_representations/' + \
                layer_name.replace("/", "_") + '.csv'
            print('\tOpening CSV file with layer representations...')
            df = pd.read_csv(layer_filename)
            print('\tDone!')
            # The names of the stimuli are the columns:
            stimuli = list(df.columns)
            triplets = []
            # Index for keeping track of the triplets:
            t = 0  # noqa
            # For each stimulus, sort them into standard/probe, shape and
            # colour matches.
            # These are taken from the stimuli on Linda Smith's website:
            # http://www.indiana.edu/~cogdev/SB_testsets.html
            for s, stimulus in enumerate(stimuli):
                # The location codes what type of stimulus we are looking at:
                location = int(stimulus.split('-')[1])
                # Prepends the role the stimulus plays and the triplet it is
                # in:
                if (location % 3) == 0:  # this is the standard item
                    stimuli[s] = 'standard-' + \
                        str(int(location / 3)) + '-' + stimuli[s]
                    triplets.append([[], [], []])
                    triplets[t][0] = stimuli[s]
                elif (location % 3) == 1:  # this is the shape match
                    stimuli[s] = 'shape_match-' + \
                        str(int(np.floor(location / 3))) + '-' + stimuli[s]
                    triplets[t][1] = stimuli[s]
                elif (location % 3) == 2:  # this is the colour match
                    stimuli[s] = 'colour_match-' + \
                        str(int(np.floor(location / 3))) + '-' + stimuli[s]
                    triplets[t][2] = stimuli[s]
                    t += 1  # index for next triplet
            # Rename the columns to reflect the roles of each stimulus and the
            # triplet they belong to:
            df.columns = stimuli
            layer_luce_accuracy = []
            layer_max_accuracy = []
            # For each triplet:
            for triplet in triplets:
                # Create a dataframe that contains the similarities required,
                # i.e., between probe and shape match and between probe and
                # colour match.
                similarity = df.loc[:, triplet]
                similarity = similarity.corr() + 1
                similarity = similarity[triplet[1:]]
                similarity = pd.DataFrame(similarity.iloc[0])
                sim_to_shape = float(similarity.iloc[0])
                sim_to_colour = float(similarity.iloc[1])
                # Luce choice accuracy:
                layer_luce_accuracy.append(
                    sim_to_shape / (sim_to_colour + sim_to_shape))
                if sim_to_shape > sim_to_colour:
                    layer_max_accuracy.append(1)
                else:
                    layer_max_accuracy.append(0)

            luce_accuracy.append(np.asarray(layer_luce_accuracy).mean())
            max_accuracy.append(np.asarray(layer_max_accuracy).mean())
        np.savetxt(EXP_DIR + 'luce_accuracy.csv', luce_accuracy)
        np.savetxt(EXP_DIR + 'max_accuracy.csv', max_accuracy)

    fig, ax = plt.subplots()

    ax.plot(max_accuracy, marker='o')
    ax.set_xlim([-1, 26])
    ax.set_ylim([0.25, 0.85])

    ax.set_xlabel('Network Layer', size=20)
    ax.set_ylabel('Shape Choice', size=20)
#     ax.legend(loc=2)
    ax.axvline(x=12.5, color='#000000',
                linestyle='solid', alpha=0.1, linewidth=10)

    # We define a fake sub-plot that is in fact the only plot.
    # plot = fig.add_subplot(111)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    sns.despine(offset=10, trim=True)

    fig.savefig(EXP_DIR + 'max_cog_dev_shape_match_line.pdf',
                format='pdf', bbox_inches='tight')

    return EXP_DIR + 'max_cog_dev_shape_match_line.pdf'


if __name__ == '__main__':
    create_and_save_figures()
