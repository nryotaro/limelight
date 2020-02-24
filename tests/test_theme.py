from unittest import TestCase
import numpy as np
import numpy.testing as npt
import limelight.theme as t


class TestTheme(TestCase):

    def test_create(self):
        actual = t.Theme.create('talk.politics.mideast')
        self.assertEqual(actual, t.Theme.TALK_POLITICS_MIDEAST)

    def test_get_theme_list(self):
        actual = t.Theme.get_themename_list()
        self.assertEqual(len(actual), 20)
        self.assertIn('comp.os.ms-windows.misc', actual)


class TestThemes(TestCase):

    def test_get_index_matrix_mono(self):
        themes = t.Themes([t.Theme.TALK_POLITICS_GUNS])

        actual = themes.get_index_matrix()
        expected = np.zeros([1, 20])
        expected[0, 15] = 1
        npt.assert_equal(
            actual,
            expected)

    def test_get_index_matrix_multi(self):
        themes = t.Themes([
            t.Theme.TALK_POLITICS_GUNS,
            t.Theme.SCI_MED,
        ])

        actual = themes.get_index_matrix()
        expected = np.zeros([2, 20])
        expected[0, 15] = 1
        expected[1, 8] = 1
        npt.assert_equal(
            actual,
            expected)
