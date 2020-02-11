from unittest import TestCase
import limelight.theme as t


class TestTheme(TestCase):

    def test_create(self):
        actual = t.Theme.create('talk.politics.mideast')
        self.assertEqual(actual, t.Theme.TALK_POLITICS_MIDEAST)

    def test_get_theme_list(self):
        actual = t.Theme.get_themename_list()
        self.assertEqual(len(actual), 20)
        self.assertIn('comp.os.ms-windows.misc', actual)
