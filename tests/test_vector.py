from unittest import TestCase
import numpy as np
import numpy.testing as npt
import limelight.vector as v


class TestDenseTextVectors(TestCase):

    def test_raw(self):
        vectors = np.zeros([1, 2])
        target = v.DenseTextVectors(vectors)

        npt.assert_array_equal(target.raw(), vectors,
                               'raw() returns the passed dence_vectors.')
