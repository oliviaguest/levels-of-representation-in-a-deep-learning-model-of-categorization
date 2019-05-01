# Cardiogram Experiment

## Exemplar Models
The exemplar models are trained and tested on the layer representations of the stimulus sets (```circle_square``` and ```circle_square_bounding_box```). ```circle_square``` and ```circle_square_bounding_box``` are further divided into : ```colour``` (red and blue), ```grayscale``` (light gray and dark gray). The ```results``` and ```accuracy``` directories contain the details for the exemplar models.

## Stimuli
Stimuli for the deep network are stored in ```stimuli```.

## Figures
The ```figures``` directory contains the accuracy figures for models trained on each stimulus set. Generated using ```figures.py```.

## Layer Representations
The deep network's layer states for each input stimulus are stored in ```layer_representations``` with subdirectories for the two types of stimuli (```circle_square``` and ```circle_square_bounding_box``` and deeper still ```colour``` and ```grayscale```).

## Code

### ```net.py```
Run deep network on the stimuli, saved outputs to ```layer_representations``` directory. Does not need to be run if the required layer representations are present.

### ```misc.py```
Functions and constants shared between ```exemplar_model_circle_square.py``` and ```exemplar_model_circle_square_bounding_box.py```.

###  ```exemplar_model_circle_square.py```
Run the exemplar model trained on the stimuli in ```./stimuli/circle_square```.

### ```exemplar_model_circle_square_bounding_box.py```.
Run the exemplar model trained on the stimuli in ```./stimuli/circle_square_bounding_box```.

### ```figures.py```
Create the accuracy figures for both exemplar models.
