"""Figures for the journal article."""
from __future__ import print_function

import os
import gabor_experiment.sim
import gabor_experiment.figure
import cog_dev_experiment.accuracy
import shapes_experiment.paper_figures
import cardiogram_experiment.paper_figure_accuracies
import cardiogram_experiment.paper_figure_layer_0

# Run all the figures:
FIGURES = [shapes_experiment.paper_figures.create_and_save_figures(),
           gabor_experiment.sim.create_and_save_figures(),
           gabor_experiment.figure.create_and_save_figures(),
           cog_dev_experiment.accuracy.create_and_save_figures(),
           cardiogram_experiment.paper_figure_accuracies.
           create_and_save_figures(),
           cardiogram_experiment.paper_figure_layer_0.
           create_and_save_figures()]

ARTICLE_FIGURES_PATH = '..//Deep-Convolutional-Neural-Networks-as-Models-' + \
    'of-Categorization/fig/'

for figure in FIGURES:
    print(ARTICLE_FIGURES_PATH + os.path.basename(figure))
    os.system('pdf-crop-margins -s -u ' + figure + ' -o '
              + ARTICLE_FIGURES_PATH + os.path.basename(figure))
