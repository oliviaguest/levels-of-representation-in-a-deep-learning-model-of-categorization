# cnn-vs-pigeons

[![Build Status](https://travis-ci.com/oliviaguest/cnn-vs-pigeons.svg?token=qe5169Xpv1Woxy1aGgNr&branch=master)](https://travis-ci.com/oliviaguest/cnn-vs-pigeons)

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
python shapes_experiment/exemplar_model_circle_square_bounding_box.py.
```
For more information, see ```README.md``` in directory ```shapes_experiment```.

## Shape Bias (CogDev Stimuli) Experiment
Run scripts from this directory, e.g.:
```
python cog_dev_experiment/exemplar_model.py.
```
For more information, see ```README.md``` in directory ```cog_dev_experiment```.

## Gabor Patches Experiment
Run scripts from this directory, e.g.:
```
python gabor_experiment/pca.py.
```
For more information, see ```README.md``` in directory ```gabor_experiment```.


# Ideas for Future Consideration

## Ideas from ASIC
Try randomising the weights, the pixels, and/or the activations on each layer idea based on suggestion by Roger Ratcliff

## Ideas from P01 meeting
Sometimes there might be a colour bias in deeper layers because it's important for categorisation. e.g., birds which have red breast vs yellow.

## Ideas from Brad and Lab Meeting
* Try Gabor patches to see where orientation arises in the network.
* How does the cardiogram machine work? Does it explicitly use pixel space?
* Try network on images it knows rotated through to check how orientation affects accuracy.

# Journal Article
## Brad
For action editor at JEP:General, we want Andrew Conway as he handled the Nosofsky rock paper. We should also cite that paper which will be in press soon as dealing with natural stimuli (using ratings and such) and perhaps request Rob Nosofsky as a reviewer (maybe not as he can be tough).
I might as well send the neural network analysis stuff.

(1) get an 8100 (each item) x number of units matrix
(2) Do a PCA.
(3) Plot the amount of variance explained for different number of components included.
(4) create two vectors of 8100 dimensions.
(a) one vector is the orientation value of the stimulus orientation (1-90).
(b) the other is width (1-90)
(5) take one of these vectors and correlate with one of the two first principal component for a 2x2 correlation matrix. E.g., if orientation vector correlations with second component, then that component represents orientation.
