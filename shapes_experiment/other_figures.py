import glob

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from shapes_experiment.misc import (ACCURACY_DIR, FIGURES_DIR)


# for df_file in df_files:
#     df = pd.read_csv(df_file, index_col=0)
#
#     df['Layer Index'] = df.index
#     df = df[['Layer Index', u'Background Stimulus',  u'Foreground Hue Type', u'Luce Accuracy',
#    u'Max Accuracy', u'Optimum Accuracy', u'Same Hue Luce Accuracy',
#    u'Same Size Luce Accuracy', u'Same Shape Luce Accuracy',
#    u'Same Hue Optimum Accuracy', u'Same Size Optimum Accuracy',
#    u'Same Shape Optimum Accuracy']]
#     df.to_csv(df_file, index=False)
def fix_df(df):
    df.reset_index(inplace=True)
    df.rename(index=int, columns={"index": "Layer Index"}, inplace=True)
    return df


def create_figure(column_name, file_name_postfix):
    sns.set(font_scale=1.6)
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    # Plot tip as a function of toal bill across days
    # g = sns.lmplot(x="Layer Index", y=column_name,
    #                hue="Background Stimulus Type",
    #                truncate=True, data=df,  palette=sns.color_palette("muted", 6), size=5,
    #                aspect=1.2,
    #                hue_order=["Clouds", "Seascape",
    #                           "Water", "Machinery", "Cars"],
    #                x_jitter=0.25,
    #                # scatter_kws={'alpha': 0.5},
    #                legend=False,
    #                legend_out=False, fit_reg=False)

    ax = sns.pointplot(ax=ax, x="Layer Index", y=column_name, hue="Hue Type",
                       data=df, markers='.', capsize=.2,
                       palette=sns.color_palette(["#999999", "#c63d92"]),
                       hue_order=["grayscale", "colour"])
    sns.despine(offset=10, trim=True)

    ax.set(xlabel='Layer', ylabel=column_name)

    # Use more informative axis labels than are provided by default
    # g.set_axis_labels("Layer", "Accuracy")
    if 'difference' in column_name.lower():
        None
    elif 'optimum' in column_name.lower():
        plt.axis([-1, 26, 0.48, 1.01])
    elif 'luce' in column_name.lower():
        plt.axis([-1, 26, 0.48, 0.7])

    # plt.legend(frameon=False, loc=4)
    # plt.legend(ncol=1, frameon=True, bbox_to_anchor=(1.05, 1),
    # borderaxespad=0.)
    # plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left",
               # borderaxespad=0., fontsize=14)

    # fig.set_size_inches(10, 6)
    for t, tick in enumerate(ax.xaxis.get_major_ticks()):
        if t % 5:
            tick.set_visible(False)
    for l, label in enumerate(ax.xaxis.get_ticklabels()):
        if l % 5:
            label.set_visible(False)
    fig.savefig(FIGURES_DIR + column_name.lower().replace(' ', '_') + file_name_postfix + '.pdf',
                bbox_inches='tight')
    fig.savefig(FIGURES_DIR + column_name.lower().replace(' ', '_') + file_name_postfix + '.png',
                bbox_inches='tight')
    plt.close(fig)

df_files = glob.glob(ACCURACY_DIR + '*_accuracy.csv')
first_files = []
second_files = []
for df_file in df_files:
    if 'box' in df_file:
        first_files.append(df_file)
    else:
        second_files.append(df_file)
df_files = [first_files, second_files]

for df_files_iter in df_files:
    try:
        del df
    except NameError:
        pass
    for df_file in df_files_iter:
        file_name_postfix = '_overlapping'
        if 'box' in df_file:
            file_name_postfix = '_bounding_box'
        try:
            df = df.append(fix_df(pd.read_csv(df_file)))  # noqa
        except NameError:
            df = fix_df(pd.read_csv(df_file))

    df.rename(columns={"Unnamed: 0": "Layer Name"}, inplace=True)

    df['Shape-hue optimum accuracy difference'] = df['Same Shape Optimum Accuracy'] - \
        df['Same Hue Optimum Accuracy']
    df['Shape-size optimum accuracy difference'] = df['Same Shape Optimum Accuracy'] - \
        df['Same Size Optimum Accuracy']
    df['Shape-hue Luce accuracy difference'] = df['Same Shape Luce Accuracy'] - \
        df['Same Hue Luce Accuracy']
    df['Shape-size Luce accuracy difference'] = df['Same Shape Luce Accuracy'] - \
        df['Same Size Luce Accuracy']

    columns = []

    for column in df.columns:
        if 'optimum' in column.lower() or 'luce' in column.lower() or 'max' in column.lower():
            # if 'difference' in column:
            columns.append(column)

    for column in columns:
        print(column)
        create_figure(column, file_name_postfix)
