"""Pruebas de métricas con datos sintéticos, no resultados del proyecto."""
import unittest
import pandas as pd
from src.evaluation.compare_models import evaluate


class EvaluationTests(unittest.TestCase):
    def test_perfect(self):
        frame = pd.DataFrame({'y_true':['good','regular','bad'], 'y_pred':['good','regular','bad']})
        self.assertEqual(evaluate(frame), {'accuracy':1.0,'f1_macro':1.0})

    def test_minority_error(self):
        frame = pd.DataFrame({'y_true':['good','good','regular','bad'], 'y_pred':['good','good','good','bad']})
        result = evaluate(frame)
        self.assertAlmostEqual(result['accuracy'], 0.75)
        self.assertAlmostEqual(result['f1_macro'], 0.6)


if __name__ == '__main__':
    unittest.main()
