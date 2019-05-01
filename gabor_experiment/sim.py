"""Non-overlapping gabor patch experiment.

Using stimuli from two "opposite" sets compare the ranks of the "same" stimulus
from one set to the other and vice versa.
"""

from __future__ import division, print_function

import os

import numpy as np
import pandas as pd
import seaborn as sns

from collections import OrderedDict

import matplotlib.pyplot as plt

from gabor_experiment.misc import (get_subset, get_representations,
                                   LEFT_STIMULI_DIR, RIGHT_STIMULI_DIR,
                                   LEFT_REPS_DIR, RIGHT_REPS_DIR, FIGURES_DIR,
                                   RESULTS_DIR)

from utils.misc import LAYER_NAMES



def create_and_save_heatmap(corr, save_filename):
    """Create and save heatmap of correlations."""
    sns.set()
    sns.set(style="whitegrid")
    sns.set(font_scale=1.6, style="ticks")
    # Correlation heatmap:
    fig, ax = plt.subplots(figsize=(11, 9))  # pylint: disable=C0103
    # Generate a custom diverging colormap:
    cmap = sns.diverging_palette(255, 0, center='light', as_cmap=True)
    # Draw the heatmap:
    sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True)
    xticks = [xticklabel[:6] for xticklabel in corr.columns]
    yticks = [yticklabel[:6] for yticklabel in corr.index][::-1]
    ax.set_xticklabels(xticks, rotation=90)
    ax.set_yticklabels(yticks, rotation=0)
    # We change the fontsize of minor ticks label
    # plt.tick_params(axis='both', which='major', labelsize=7)
    # plt.tick_params(axis='both', which='minor', labelsize=7)
    plt.xlabel(corr.columns[0][-4:].capitalize())
    plt.ylabel(corr.index[0][-5:].capitalize())

    fig.savefig(save_filename + '.pdf',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(save_filename + '.png',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def get_rank(one, two, save_dir):
    """Return rank of stimulus one when correlated with its counterpart in two.

    Take stimulus in one, e.g., with label 'f10o01_left', and open file with
    pre-computed correlations in descending order of correlation with all items
    in set two, i.e., the right stimuli. Return the rank of the stimulus in two
    which matches in name to that in one, i.e., if one is 'f10o01_left' then
    the rank returned is that of 'f10o01_right'.
    """
    tmp = pd.read_csv(save_dir + one + '.csv')
    tmp = tmp.set_index(['index'])
    tmp = tmp.rename(columns={'Unnamed: 0': 'Rank'})
    tmp.Rank += 1  # add 1 because we want ranks to start at 1
    return tmp.loc[two].Rank


def create_and_save_figures():
    """Make the figures."""
    sns.set()
    sns.set(style="whitegrid")
    sns.set(font_scale=1.6, style="ticks")
    # stimuli_dirs = [LEFT_STIMULI_DIR, RIGHT_STIMULI_DIR]
    # other_stimuli_dirs = [RIGHT_STIMULI_DIR, LEFT_STIMULI_DIR]
    # reps_dirs = [LEFT_REPS_DIR, RIGHT_REPS_DIR]
    # other_reps_dirs = [RIGHT_REPS_DIR, LEFT_REPS_DIR]

    # Get the labels for the stimuli we want to use:
    left_subset = get_subset(LEFT_STIMULI_DIR, True)
    left_subset_paths = get_representations(left_subset, LEFT_REPS_DIR)

    right_subset = get_subset(RIGHT_STIMULI_DIR, True)
    right_subset_paths = get_representations(right_subset, RIGHT_REPS_DIR)

    # Set up the dataframe:
    column = [('Image', left_subset + right_subset)]
    df = pd.DataFrame.from_dict(OrderedDict(column))  # noqa
    df = df.set_index(['Image'])  # noqa

    for l, layer_name in enumerate(LAYER_NAMES):
        save_dir = RESULTS_DIR + '/' + layer_name.replace("/", "_") + '/'
        # Some dummy values for the cells to-be-populated:
        df[str(l)] = [-100000 for _ in left_subset + right_subset]

        try:
            # If the correlation files exist!
            for left, right in zip(left_subset, right_subset):
                df.loc[left][str(l)] = get_rank(left, right, save_dir)
                df.loc[right][str(l)] = get_rank(right, left, save_dir)
        except IOError:
            # They don't, so let's create them.
            print(l, layer_name)
            paths = left_subset_paths[l] + right_subset_paths[l]
            for path in paths:
                print('path', path)
                try:
                    reps_df = pd.concat([reps_df, pd.read_csv(path)],  # noqa
                                        axis=1, join='inner')
                except NameError:
                    reps_df = pd.read_csv(path)

            corr = reps_df.corr().filter(left_subset).drop(left_subset)
            del reps_df
            save_filename_corr = FIGURES_DIR + 'heatmap_layer_' + str(l)

            create_and_save_heatmap(corr, save_filename_corr)

            # Create a heatmap of the correlations:
            orientation = np.array([int(o[4:6]) for o in corr.index])
            frequency = np.array([int(f[1:3]) for f in corr.index])
            corr['Frequency'] = frequency
            corr['Orientation'] = orientation
            alt_corr = corr.sort_values(['Orientation', 'Frequency'])
            del corr['Frequency']
            del corr['Orientation']
            new_columns = [label[:6] + alt_corr.columns[0][-5:] for label in
                           alt_corr.index]
            alt_corr = alt_corr[new_columns]
            save_filename_alt_corr = (FIGURES_DIR + 'heatmap_alt_layer_'
                                      + str(l))
            create_and_save_heatmap(alt_corr, save_filename_alt_corr)

            try:
                os.makedirs(save_dir)
            except OSError:
                pass

            for column in corr.columns:
                pd.DataFrame(
                    corr.sort_values([column], ascending=False)[column])\
                    .reset_index().to_csv(save_dir + column + '.csv')

            for column in corr.T.columns:
                pd.DataFrame(
                    corr.T.sort_values([column], ascending=False)[column])\
                    .reset_index().to_csv(save_dir + column + '.csv')

    # Create another figure to visualise these correlations:
    plot_df = pd.DataFrame(df.mean())  # noqa
    plot_df.columns = ['Mean Rank']
    # print(plot_df)
    plot_df['Layer'] = plot_df.index
    plot_df['Layer'] = plot_df['Layer'].astype('int64')

    colors = ["grass"]
    sns.set_palette(sns.xkcd_palette(colors))
    fig, ax = plt.subplots()

    ax = sns.regplot(y='Mean Rank',  # pylint: disable=C0103
                     x='Layer', data=plot_df, ax=ax)

    plt.xlim([-1, 26])
    plt.ylim([0, 60])
    ax.set_xlabel('Network Layer', size=20.25)
    ax.set_ylabel('Rank', size=20.25)
    # ax.fill([-1, -1, 26, 26], [40.5, 60, 60, 40.5], c='gray', alpha=0.1)
    sns.despine(offset=10, trim=True)

    save_filename = FIGURES_DIR + 'mean_rank_per_layer'
    fig.savefig(save_filename + '.pdf',
                bbox_inches='tight')
    fig.savefig(save_filename + '.png',
                bbox_inches='tight')

    # Here we create a "long" version of the dataframe, so we can do the OLS:
    long_df = pd.DataFrame(columns=['Rank', 'Layer', 'Stimulus'], dtype=float)

    for idx in df.index:
        for layer, rank in enumerate(df.loc[idx]):
            long_df = long_df.append(pd.DataFrame({'Rank': [rank],
                                                   'Layer': [layer],
                                                   'Stimulus': [idx]}),
                                     ignore_index=True)
    long_df['Layer'] = pd.to_numeric(long_df['Layer'])
    fig, ax = plt.subplots()  # pylint: disable=C0103
    # ax.fill([-1, -1, 26, 26], [40.5, 82, 82, 40.5], c='red', alpha=0.05)
    # ax.fill([-1, -1, 26, 26], [40.5, 0, 0, 40.5], c='#0033aa', alpha=0.035)

    ax.plot(long_df.groupby(by='Layer').mean(), color='#ff63c8', linewidth=3,
            alpha=0.6, marker='.', label='Mean rank')
    ax = sns.regplot(x=long_df["Layer"],  # pylint: disable=C0103
                     y=long_df["Rank"], x_jitter=0.1,
                     scatter_kws={'alpha': 0.1, 's': 20}, color="#0073b1",
                     ci=95)
    plt.xlim([-1, 26])
    plt.ylim([-5, 85])
    ax.set_xlabel('Network Layer', size=20.25)
    ax.set_ylabel('Rank', size=20.25)

    ax.set_yticks(np.arange(0, 85, 20))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.075), frameon=False)
    ax.tick_params(axis='both', which='both', top='off', right='off')

    sns.despine(offset=10, trim=True)

    save_filename = FIGURES_DIR + 'rank_per_layer.pdf'
    fig.savefig(save_filename, bbox_inches='tight')
    # return save_filename


    fig, ax = plt.subplots()  # pylint: disable=C0103
    # ax.fill([-1, -1, 26, 26], [40.5, 82, 82, 40.5], c='red', alpha=0.05)
    # ax.fill([-1, -1, 26, 26], [40.5, 0, 0, 40.5], c='#0033aa', alpha=0.035)

    ax.plot(100 - 100* long_df.groupby(by='Layer').mean().values /81, color='#1f78b4',
            linewidth=3, alpha = 0.6, label='Mean rank')

    # ax.errorbar(range(0,26), 100 - 100* long_df.groupby(by='Layer').mean().values /81,
    #             color='#ff63c8',

    #             alpha = 0.6,
    #             yerr=100* long_df.groupby(by='Layer').std().values /81,
    #             fmt='', ecolor='#cccccc', capthick=1
    #            )
    100* long_df.groupby(by='Layer').std()/81

    # ax = sns.regplot(x=long_df["Layer"],  # pylint: disable=C0103
    #                  y=long_df["Mean Rank Percentile"], x_jitter=0.1,
    #                  scatter_kws={'alpha': 0.1, 's': 20}, color="#0073b1",
    #                  ci=95
    #                 )
    plt.xlim([-1, 26])
    plt.ylim([35, 110])
    ax.set_xlabel('Network Layer', size=20.25)
    ax.set_ylabel('Mean Rank Percentile', size=20.25)
    ax.tick_params(axis='both', which='both', top='off', right='off')

    # ax.legend(loc="upper left", bbox_to_anchor=(0,1.05))
    ax.set_yticks(np.arange(40, 110, 10))

    sns.despine(offset=10, trim=True)

    save_filename = FIGURES_DIR + 'mean_rank_per_layer.pdf'
    fig.savefig(save_filename, bbox_inches='tight')
    return save_filename


if __name__ == '__main__':
    create_and_save_figures()
