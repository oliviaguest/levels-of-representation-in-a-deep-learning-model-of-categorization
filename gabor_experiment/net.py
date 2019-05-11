#!/usr/bin/env python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# Modified by Olivia Guest.
#

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import division, print_function

import os
import os.path

import pandas as pd
import tensorflow as tf

from gabor_experiment.misc import (ORIGINAL_STIMULI_DIR, ORIGINAL_REPS_DIR,
                                   LEFT_STIMULI_DIR, LEFT_REPS_DIR,
                                   RIGHT_STIMULI_DIR, RIGHT_REPS_DIR,
                                   get_subset)


from utils.misc import LAYER_NAMES
from utils.iv3 import (maybe_download_and_extract,
                       create_graph)


def run_on_images_and_save_as_dfs(images, save_dir):
    """Create graph from saved GraphDef."""
    create_graph()
    postfix = ':0'

    sess = tf.Session()
    labels = [os.path.splitext(os.path.basename(image))[0] for image in images]
    # Get every state (per requested layer, see: LAYER_NAMES) for every image:
    layer_tensors = []
    for layer_name in LAYER_NAMES:
        layer_tensors.append(
            sess.graph.get_tensor_by_name(layer_name + postfix))

    # For each layer:
    for l, layer_tensor in enumerate(layer_tensors):
        layer_activations = []
        # And for each image:
        for i, image in enumerate(images):
            # Get the raw image:
            image_data = tf.gfile.FastGFile(image, 'rb').read()
            layer_activations = sess.run(layer_tensor,
                                         {'DecodeJpeg/contents:0': image_data})
            layer_activations = layer_activations.flatten()
            print(i, image, LAYER_NAMES[l], layer_activations.shape)

            column = [(labels[i], layer_activations)]
            df = pd.DataFrame.from_items(column)

            try:
                os.mkdir(save_dir + LAYER_NAMES[l].replace("/", "_"))
            except OSError:
                None
            df.to_csv(save_dir + LAYER_NAMES[l].replace("/", "_")
                      + '/' + labels[i] + '.csv', index=False)


def main(_):
    """Main function that calls all others."""
    maybe_download_and_extract()
    import shutil
    stimuli_dirs = [ORIGINAL_STIMULI_DIR, LEFT_STIMULI_DIR, RIGHT_STIMULI_DIR]
    reps_dirs = [ORIGINAL_REPS_DIR, LEFT_REPS_DIR, RIGHT_REPS_DIR]

    for stimuli_dir, reps_dir in zip(stimuli_dirs, reps_dirs):
        subset = get_subset(stimuli_dir)
        # print(subset)
        # for f in subset:
        #     shutil.copy(f, '../Deep-Convolutional-Neural-Networks-as-Models-of-Categorization/fig/')
        # exit()
        try:
            os.makedirs(reps_dir)
        except OSError:
            pass
        run_on_images_and_save_as_dfs(subset, reps_dir)


if __name__ == '__main__':
    tf.app.run()
