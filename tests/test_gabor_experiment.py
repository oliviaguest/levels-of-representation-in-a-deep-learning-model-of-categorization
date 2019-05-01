"""Testing."""

from gabor_experiment.misc import get_subset
from gabor_experiment.misc import LEFT_STIMULI_DIR
from gabor_experiment.misc import ORIGINAL_REPS_DIR
from gabor_experiment.misc import ORIGINAL_STIMULI_DIR
from gabor_experiment.misc import RIGHT_STIMULI_DIR
from gabor_experiment.misc import STIMULI_POSTFIXES
import gabor_experiment.pca as pca
import os
import unittest
from utils.misc import LAYER_NAMES


class TestGaborExperimentPCA(unittest.TestCase):  # noqa:D101
    def test_run_and_plot_pca(self):  # noqa:D102
        # I specifically added f10o01 and f10o11 to git so tests run on server
        # too:
        subset = ['f10o01', 'f10o11']
        layer_index = 25
        layer_name = LAYER_NAMES[layer_index]

        # Collect up the paths to these stimuli's representations on each
        # layer:
        subset_paths = []
        for stimulus in subset:
            subset_paths.append(ORIGINAL_REPS_DIR +
                                layer_name.replace("/", "_") +
                                '/' + stimulus + '.csv')

        transformed_x = pca.run_and_plot_pca(subset_paths,
                                             './tests/figures/gabor_experiment'
                                             '_run_and_plot_pca')

        # Must have the same number of dimensions as inputs:
        self.assertEqual(len(transformed_x), len(subset))


class TestGaborExperimentMisc(unittest.TestCase):  # noqa:D102
    @unittest.skipIf(len(os.listdir(ORIGINAL_STIMULI_DIR)) <= 81,
                     "skip if the files don't exist")
    def test_get_subset(self):  # noqa:D101
        stimuli_dirs = [ORIGINAL_STIMULI_DIR,
                        LEFT_STIMULI_DIR,
                        RIGHT_STIMULI_DIR]

        for stimuli_dir, stimuli_postfix in zip(stimuli_dirs,
                                                STIMULI_POSTFIXES):
            self.assertEqual(len(get_subset(stimuli_dir)), 81)
            self.assertEqual(len(get_subset(stimuli_dir, True)), 81)
            stimuli_names = get_subset(stimuli_dir, True)

            for name in stimuli_names:
                self.assertTrue(stimuli_postfix in name)


if __name__ == '__main__':
    unittest.main()
