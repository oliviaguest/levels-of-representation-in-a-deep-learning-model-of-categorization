# Levels of Representation in a Deep Learning Model of Categorization

[![Build Status](https://travis-ci.org/oliviaguest/levels-of-representation-in-a-deep-learning-model-of-categorization.svg?branch=master)](https://travis-ci.org/oliviaguest/levels-of-representation-in-a-deep-learning-model-of-categorization) ![Twitter Follow](https://img.shields.io/twitter/follow/o_guest.svg?style=social)

# Preprint
[Levels of Representation in a Deep Learning Model of Categorization](https://www.biorxiv.org/content/10.1101/626374v1)
[Olivia Guest](//oliviaguest.com), [Bradley C. Love](//bradlove.org)
bioRxiv 626374; doi: https://doi.org/10.1101/626374

# Data
Please download all data and merge into the same directory structure from the OSF repo: [https://osf.io/jxavn/](https://osf.io/jxavn/).

# Experiments
## Cardiogram Images Experiment
Run scripts from this directory, e.g.:
```
python cardiogram_experiment/exemplar_model.py colour colour
```
For more information, see ```README.md``` in directory ```cardiogram_experiment```.

## 2D Shapes Experiment (Circles and Squares)
Run scripts from this directory, e.g.:
```
python shapes_experiment/exemplar_model_circle_square_bounding_box.py
```
For more information, see ```README.md``` in directory ```shapes_experiment```.

## Shape Bias (CogDev Stimuli) Experiment
Run scripts from this directory, e.g.:
```
python cog_dev_experiment/exemplar_model.py
```
For more information, see ```README.md``` in directory ```cog_dev_experiment```.

## Gabor Patches Experiment
Run scripts from this directory, e.g.:
```
python gabor_experiment/pca.py
```
For more information, see ```README.md``` in directory ```gabor_experiment```.
