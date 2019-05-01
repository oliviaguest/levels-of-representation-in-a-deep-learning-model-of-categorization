# Gabor Experiment

## Stimuli
Gabor patches with properties encoded in the filename:
```
f11o45
```
where the ```f11``` denotes the frequency and the ```o45``` the orientation.

## Code

### ```net.py```
Run deep network on the stimuli, saved outputs to ```layer_representations``` directory. Does not need to be run if the required layer representations are present.

### ```pca.py```
Do the bulk of the following items:
1. get an N (each item) x number of units matrix.
2. Do a PCA.
3. Plot the amount of variance explained for different number of components
    included.
4. create two vectors of N dimensions.
   a. one vector is the orientation value of the stimulus orientation (1-90).
   b. the other is width (1-90).
5. take one of these vectors and correlate with one of the two first principal
    component for a 2x2 correlation matrix. E.g., if orientation vector
    correlations with second component, then that component represents
    orientation.

### ```sim.py```
Non-overlapping gabor patch experiment. Using stimuli from two "opposite" sets compare the ranks of the "same" stimulus from one set to the other and vice versa. By opposite here, I mean they are placed on non-overlapping parts of the input pixel space.

### ```misc.py```
Functions and constants shared between ```pca.py``` and ```sim.py```.

## To do 
Plot the top two correlations for each layer on results of ```pca.py```.
