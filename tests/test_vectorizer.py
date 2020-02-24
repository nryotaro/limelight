from unittest import TestCase
from unittest.mock import MagicMock, patch
import sklearn.feature_selection as s
import limelight.vectorizer as v


class TestVectorizer(TestCase):

    @patch('joblib.load')
    def test_load(self, load):
        filename = MagicMock(spec=str)
        actual = v.Vectorizer.load(filename)
        expected = load.return_value
        self.assertEqual(actual, expected, 'load is a classmethod.')


class TestFeatureSelectedVectorizer(TestVectorizer):

    def test_instantiation(self):
        vectorizer = MagicMock(spec=v.Vectorizer)
        select_from_model = MagicMock(spec=s.SelectFromModel)
        v.FeatureSelectedVectorizer(
            vectorizer, select_from_model)
