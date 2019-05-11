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

### ```pca_permutation.py```
1. Get an N (each item) x number of units matrix.
2. Do a PCA.
3. Plot the amount of variance explained for different number of components
    included.
4. Create two vectors of N dimensions. One vector is the orientation value of the stimulus orientation, the other is width.
5. take one of these vectors and correlate with one of the two first principal components for a 2x2 correlation matrix. E.g., if orientation vector correlations with second component, then that component represents orientation.
6. Do a permutation test on the correlations.

### ```sim.py```
Non-overlapping gabor patch experiment. Using stimuli from two "opposite" sets compare the ranks of the "same" stimulus from one set to the other and vice versa. By opposite here, I mean they are placed on non-overlapping parts of the input pixel space.

### ```misc.py```
Functions and constants shared between ```pca.py``` and ```sim.py```.

## To do
Plot the top two correlations for each layer on results of ```pca.py```.
