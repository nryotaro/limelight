from unittest import TestCase
import limelight.theme as t


class TestTheme(TestCase):

    def test_create(self):
        actual = t.Theme.create('talk.politics.mideast')
        self.assertEqual(actual, t.Theme.TALK_POLITICS_MIDEAST)
