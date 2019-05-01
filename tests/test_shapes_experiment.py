"""Testing."""

import shapes_experiment.exemplar_model_circle_square
import shapes_experiment.exemplar_model_circle_square_bounding_box
import unittest


class TestExemplarModelCircleSquare(unittest.TestCase):  # noqa:D101
    def test_create_all_categories(self):  # noqa:D102
        items = ['blue_circle_big', 'blue_circle_small',
                 'blue_square_big', 'blue_square_small',
                 'red_circle_big', 'red_circle_small',
                 'red_square_big', 'red_square_small']
        # Create all category combinations:
        prototype_as, items_as, prototype_bs, items_bs = \
            shapes_experiment.exemplar_model_circle_square. \
            create_all_categories(items)
        self.assertEqual(len(prototype_as), len(prototype_bs))
        self.assertEqual(len(items_as), len(items_bs))
        self.assertEqual(set(prototype_as).intersection(set(prototype_bs)),
                         set([]))


class TestExemplarModelCircleSquareBoundingBox(unittest.TestCase):  # noqa:D102
    def test_create_all_categories(self):  # noqa:D102
        items = ['blue_circle_big', 'blue_circle_small', 'blue_square_big',
                 'blue_square_small', 'red_circle_big', 'red_circle_small',
                 'red_square_big', 'red_square_small',
                 'prototype_blue_circle_big_left',
                 'prototype_blue_circle_big_right',
                 'prototype_blue_circle_small_left',
                 'prototype_blue_circle_small_right',
                 'prototype_blue_square_big_left',
                 'prototype_blue_square_big_right',
                 'prototype_blue_square_small_left',
                 'prototype_blue_square_small_right',
                 'prototype_red_circle_big_left',
                 'prototype_red_circle_big_right',
                 'prototype_red_circle_small_left',
                 'prototype_red_circle_small_right',
                 'prototype_red_square_big_left',
                 'prototype_red_square_big_right',
                 'prototype_red_square_small_left',
                 'prototype_red_square_small_right']
        # Create all category combinations:
        prototype_as, items_as, prototype_bs, items_bs = \
            shapes_experiment.exemplar_model_circle_square_bounding_box. \
            create_all_categories(items)

        self.assertEqual(len(prototype_as), len(prototype_bs))
        self.assertEqual(len(items_as), len(items_bs))
        self.assertEqual(set(prototype_as).intersection(set(prototype_bs)),
                         set([]))
        self.assertTrue(shapes_experiment.
                        exemplar_model_circle_square_bounding_box.
                        prototypes_are_opposite(prototype_as, prototype_bs))
        self.assertTrue(shapes_experiment.
                        exemplar_model_circle_square_bounding_box.
                        categories_are_opposite(items_as, items_bs))


if __name__ == '__main__':
    unittest.main()
