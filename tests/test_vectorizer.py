from unittest import TestCase
from unittest.mock import MagicMock, patch
import sklearn.feature_selection as s
import sklearn.linear_model as li
import limelight.vectorizer as v


class TestVectorizer(TestCase):

    @patch('joblib.load')
    def test_load(self, load):
        filename = MagicMock(spec=str)
        actual = v.Vectorizer.load(filename)
        expected = load.return_value
        self.assertEqual(actual, expected, 'load is a classmethod.')


class TestLogisticRegressionFsVectorizer(TestVectorizer):

    def test_instantiation(self):
        vectorizer = MagicMock(spec=v.Vectorizer)
        select_from_model = MagicMock(spec=s.SelectFromModel)
        v.LogisticRegressionFsVectorizer(
            vectorizer, select_from_model)

    def test_create_from_estimator(self):
        vectorizer = MagicMock(spec=v.Vectorizer)
        estimator = li.LogisticRegression()
        max_features = 20000

        actual = v.LogisticRegressionFsVectorizer.create_from_estimator(
            estimator, vectorizer, max_features)

        self.assertIsInstance(
            actual,
            v.LogisticRegressionFsVectorizer,
            'The class of the returned object is same as that of callee.')

        self.assertEqual(
            actual.select_from_model.max_features,
            max_features,
            'The third argument constraints the number of the features.')
