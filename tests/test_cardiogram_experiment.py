"""Testing."""
import cardiogram_experiment.misc as misc
import numpy as np
import pandas as pd
import unittest


class TestCardiogramExperimentMisc(unittest.TestCase):  # noqa:D101
    def test_get_stimuli_directories(self):  # noqa: D102
        # Check spelling variants give identical output:
        self.assertEqual(misc.get_stimuli_directories('color'),
                         misc.get_stimuli_directories('colour'))

        # Identical directory structure to files shared with us from Ed's lab:
        self.assertEqual(
            misc.get_stimuli_directories('color'), ['Pretraining',
                                                    'Set_A/Abnormal',
                                                    'Set_B/Abnormal',
                                                    'Set_A/Normal',
                                                    'Set_B/Normal'])
        # Identical directory structure to files shared with us from Ed's lab:
        self.assertEqual(misc.get_stimuli_directories('gray'),
                         ['Grayscale_Stimuli'])

        # These are different images, even though similar results:
        self.assertTrue(misc.get_stimuli_directories('gray') !=
                        misc.get_stimuli_directories('grayscale'))

        # My structure before I had access to 'gray':
        self.assertEqual(misc.get_stimuli_directories('grayscale'),
                         ['Pretraining_grayscale',
                          'Set_A_grayscale/Abnormal',
                          'Set_B_grayscale/Abnormal',
                          'Set_A_grayscale/Normal',
                          'Set_B_grayscale/Normal'])

    def test_is_normal(self):  # noqa:D102
        # Stimuli described in: ./cardiogram_experiment/stimuli/README.md
        # Even though Ed's lab notes describe the filename as having the
        # character A (for abnormal) or N (for normal) in a specific location
        # in the string, nonetheless it's an inapropriate way of finding out
        # category an item belongs to. For two main reasons: the A/N is not
        # always in the same location in the string (see filenames in:
        # ./cardiogram_experiment/stimuli/Pretraining); and the character A/N
        # can actually appear in the filename at other locations (see stimuli
        # filenames). Thus I instead use the perfusion damage, which is the
        # real value used to determine category membership anyway. It appears
        # at different locations in the string but there is no chance for it
        # to appear at more than one location with different meanings, like
        # with A or N, previously.

        # So as per above, these examples contain the strings that are normal:
        self.assertTrue(misc.is_normal('00'))
        self.assertTrue(misc.is_normal('01'))
        self.assertTrue(misc.is_normal('02'))
        self.assertTrue(misc.is_normal('03'))

        # These do not:
        self.assertFalse(misc.is_normal('0'))
        self.assertFalse(misc.is_normal('1'))
        self.assertFalse(misc.is_normal('2'))
        self.assertFalse(misc.is_normal('3'))

        # These examples contain the strings that are abnormal:
        abnormal = [str(i).zfill(2) for i in range(4, 52)]
        for a in abnormal:
            self.assertFalse(misc.is_normal(a))

        # NOTE: is_normal basically acts as a normal detector, everything else
        # it will treat as abnormal. Importantly, we only have stimuli from
        # '00' to '51' anyway.

    def test_prob_correct(self):  # noqa:D102
        # Create a test dataframe:
        df = pd.DataFrame.from_dict(
            {'Probability Healthy': [0.5, 0.1, 0.0, 0.8],
             'Category': ['Normal', 'Abnormal', 'Normal', 'Abnormal']})

        # Apply the function:
        df = df.apply(misc.prob_correct, axis=1)

        probability_correct = [0.5, 0.9, 0.0, 0.2]
        for i, p in enumerate(df[['Probability Correct']].values):
            # Floating point needs rounding!
            self.assertEqual(np.round(p[0], 4),
                             np.round(probability_correct[i], 4))

    def test_calculate_performance(self):  # noqa:D102
        # Create a test dataframe (this is a pretty bad model, low accuracy):
        df = pd.DataFrame.from_dict(
            {'Probability Healthy': [0.0, 0.5, 0.6, 0.5],
             'Category': ['Normal', 'Normal', 'Abnormal', 'Abnormal'],
             'Image Names': ['N_00_TEST_XXX_test', 'N_02_TEST_XXX_test',
                             'A_30_TEST_XXX_test', 'A_20_TEST_XXX_test']})
        # Set the boundary:
        test_boundary = 0.5

        true_positive, true_negative, false_positive, false_negative = \
            misc.calculate_performance(df, test_boundary)
        self.assertEqual([true_positive, true_negative,
                          false_positive, false_negative],
                         [1, 0, 1, 2])

        # Create a test dataframe (this is a perfect model):
        df = pd.DataFrame.from_dict(
            {'Probability Healthy': [0.9, 0.8, 0.2, 0.5],
             'Category': ['Normal', 'Normal', 'Abnormal', 'Abnormal'],
             'Image Names': ['N_00_TEST_XXX_test', 'N_02_TEST_XXX_test',
                             'A_30_TEST_XXX_test', 'A_20_TEST_XXX_test']})
        test_boundary = 0.6

        true_positive, true_negative, false_positive, false_negative = \
            misc.calculate_performance(df, test_boundary)
        self.assertEqual([true_positive, true_negative,
                          false_positive, false_negative],
                         [2, 2, 0, 0])

    def test_get_optimum_accuracy_boundary(self):  # noqa:D102
        # Create a test dataframe (this is a pretty bad model, low accuracy):
        df = pd.DataFrame.from_dict(
            {'Probability Healthy': [0.0, 0.5, 0.6, 0.5],
             'Category': ['Normal', 'Abnormal', 'Normal', 'Abnormal'],
             'Image Names': ['N_00_TEST_ONE_test', 'A_34_TEST_ONE_test',
                             'N_01_TEST_ONE_test', 'A_40_TEST_ONE_test']})

        accuracy, boundary = misc.get_optimum_accuracy_boundary(df)
        # Due to the randomness of the get_optimum_accuracy_boundary()
        # function the accuracy will be 0.
        try:
            self.assertEqual(accuracy, 0.25)
        except AssertionError:
            try:
                self.assertEqual(accuracy, 0.5)
            except AssertionError:
                self.assertEqual(accuracy, 0.75)

        # Nonetheless, the boundary is always between 0.5 and 0.6:
        self.assertEqual(boundary, 0.55)

        # Same as above, but dataframe in a different order:
        df = pd.DataFrame.from_dict(
            {'Probability Healthy': [0.5, 0.5, 0.6, 0.0],
             'Category': ['Abnormal', 'Abnormal', 'Normal', 'Normal'],
             'Image Names': ['A_34_XXXX_XXX_test', 'A_40_XXXX_XXX_test',
                             'N_01_XXXX_XXX_test', 'N_00_XXXX_XXX_test']})
        accuracy, boundary = misc.get_optimum_accuracy_boundary(df)
        try:
            self.assertEqual(accuracy, 0.25)
        except AssertionError:
            try:
                self.assertEqual(accuracy, 0.5)
            except AssertionError:
                self.assertEqual(accuracy, 0.75)
        self.assertEqual(boundary, 0.55)

        # Create a test dataframe (perfect model):
        df = pd.DataFrame.from_dict(
            {'Probability Healthy': [0.2, 0.5, 0.9, 0.8],
             'Category': ['Abnormal', 'Abnormal', 'Normal', 'Normal'],
             'Image Names': ['A_40_XXXX_XXX_test', 'A_34_XXXX_XXX_test',
                             'N_01_XXXX_XXX_test', 'N_00_XXXX_XXX_test']})

        accuracy, boundary = misc.get_optimum_accuracy_boundary(df)

        # Perfect model:
        self.assertEqual(accuracy, 1.0)

        # Lower and upper boundary values:
        self.assertTrue(boundary > (0.2 + 0.8) / 2)
        self.assertTrue(boundary < (0.5 + 0.9) / 2)


if __name__ == '__main__':
    unittest.main()
