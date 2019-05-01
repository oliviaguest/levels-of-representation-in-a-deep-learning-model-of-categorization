# Cardiogram Experiment

## Exemplar Models
The three directories named ```exemplar model_train_X_test_Y``` contain all the details for exemplar models trained and tested on the named layer representations of the stimulus sets. ```X``` and ```Y``` can have the following values: ```colour``` (the colourised cardiac stimuli), ```grayscale``` (the grayscale images I created based on the colourised ones, because Ed's group originally didn't send these to us), and then ```gray``` (the ones Ed's group gave us, the same ones the pigeons saw).  For all intents and purposes and I have checked this, the two grayscale stimuli are the same.

## Stimuli
Stimuli for the deep network are stored in ```stimuli```.

## Figures
Figures specific to each model are stored within the model's directory, while general figures comparing across model are in ```figures```.

## Layer Representations
The deep network's layer states for each input stimulus are stored in ```layer_representations_X``` where ```X``` is as above the stimulus set name.

## Code

### ```net.py```
Run deep network on the stimuli, saved outputs to appropriately suffixed ```layer_representations``` directory. Does not need to be run if the required layer representations are present.

### ```exemplar_model.py```
Runs and graphs the exemplar models for each layer and for each stimulus set. Saves the results for quick access if required.

### ```graph_accuracies.py```
Uses the outputs from ```exemplar_model.py``` (assuming all appropriate combinations of train and test have been run and saved) and creates figures comparing the three models.

### ```graph_layer_0.py```
Uses the outputs from ```exemplar_model.py``` (assuming all appropriate combinations of train and test have been run and saved) and creates figures comparing the three models specifically at layer 0 (the pixel space).

### ```create_grayscale_images.py```
One off script to take the colourised images and turn them into grayscale. This is how I create the ```grayscale``` stimuli.  Needs updating because directory structure has changed. Currently not in use anyway.
