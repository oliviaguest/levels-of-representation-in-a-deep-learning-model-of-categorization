import numpy as np

# Layers names based on TensorBoard graph
LAYER_NAMES = ['DecodeJpeg', 'Cast', 'ExpandDims', 'ResizeBilinear', 'Sub',
               'Mul', 'conv', 'conv_1', 'conv_2', 'pool', 'conv_3', 'conv_4',
               'pool_1', 'mixed/join', 'mixed_1/join', 'mixed_2/join',
               'mixed_3/join', 'mixed_4/join', 'mixed_5/join', 'mixed_6/join',
               'mixed_7/join', 'mixed_8/join', 'mixed_9/join', 'mixed_10/join',
               'pool_3', 'softmax']


def make_2d(array):
    dims = np.asarray(array).shape
    dim2 = 1
    for dim in dims[1:]:
        dim2 *= dim
    array = np.asarray(array)
    return array.reshape((dims[0], dim2))
