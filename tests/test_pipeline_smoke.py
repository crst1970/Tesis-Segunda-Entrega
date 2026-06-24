import unittest

import numpy as np

from script import pipeline_abide


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals = np.random.default_rng(42).normal(size=(40, 6))

    def test_main_connectivity_methods(self):
        for method, symmetric in [
            ("pearson", True),
            ("graphical_lasso", True),
            ("lingam", False),
        ]:
            matrix = pipeline_abide.compute_matrix(
                self.signals,
                method,
                maxlag=1,
                tau_max=1,
                graphical_lasso_alpha=0.5,
            )
            self.assertEqual(matrix.shape, (6, 6))
            self.assertTrue(np.isfinite(matrix).all())
            if symmetric:
                self.assertTrue(np.allclose(matrix, matrix.T, atol=1e-6))
                self.assertTrue(np.allclose(np.diag(matrix), 1.0))
            else:
                self.assertTrue(np.allclose(np.diag(matrix), 0.0))

    def test_experimental_granger_interface(self):
        matrix, selected_values = pipeline_abide.granger(
            self.signals,
            maxlag=1,
            lag_strategy="min_q",
        )
        self.assertEqual(matrix.shape, (6, 6))
        self.assertEqual(selected_values.shape, (6, 6))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(np.allclose(np.diag(matrix), 0.0))

    def test_feature_dimensions(self):
        symmetric = np.eye(6)
        directed = np.zeros((6, 6))
        self.assertEqual(
            pipeline_abide.vectorizar(symmetric, simetrica=True).shape,
            (15,),
        )
        self.assertEqual(
            pipeline_abide.vectorizar(directed, simetrica=False).shape,
            (30,),
        )


if __name__ == "__main__":
    unittest.main()
