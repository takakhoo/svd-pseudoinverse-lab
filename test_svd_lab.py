import unittest
import numpy as np
from svd_lab import pseudoinverse, penrose_residuals, experiment


class SVDTests(unittest.TestCase):
    def test_rectangular_and_complex(self):
        rng = np.random.default_rng(23)
        for shape in [(4, 4), (12, 3), (3, 12)]:
            for complex_input in [False, True]:
                a = rng.normal(size=shape)
                if complex_input:
                    a = a + 1j*rng.normal(size=shape)
                inverse = pseudoinverse(a)
                np.testing.assert_allclose(inverse, np.linalg.pinv(a), atol=1e-12)
                self.assertLess(max(penrose_residuals(a, inverse).values()), 1e-12)

    def test_rank_deficient(self):
        a = np.outer([1., 2., 3.], [4., 5.])
        np.testing.assert_allclose(pseudoinverse(a), np.linalg.pinv(a), atol=1e-14)
        self.assertLess(max(penrose_residuals(a, pseudoinverse(a)).values()), 1e-12)

    def test_zero(self):
        np.testing.assert_array_equal(pseudoinverse(np.zeros((3, 5))), np.zeros((5, 3)))

    def test_tolerance_discards_small_singular_values(self):
        a = np.diag([1., 1e-12])
        np.testing.assert_allclose(pseudoinverse(a, 1e-10), np.diag([1., 0.]))

    def test_invalid_inputs(self):
        for a, cutoff in [([1, 2], 1e-15), (np.zeros((0, 4)), 1e-15), ([[np.nan]], 1e-15), ([[1]], -1)]:
            with self.assertRaises(ValueError):
                pseudoinverse(a, cutoff)

    def test_noise_amplification_and_reproducibility(self):
        first, _ = experiment(7)
        second, _ = experiment(7)
        self.assertEqual(first, second)
        self.assertGreater(first['sweep'][0]['relative_solution_error'], 1e3)
        self.assertLess(first['sweep'][22]['relative_solution_error'], 2)


if __name__ == "__main__":
    unittest.main()
