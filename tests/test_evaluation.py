"""Pruebas de métricas con datos sintéticos, no resultados del proyecto."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from src.evaluation.compare_models import compare, evaluate, plot_history, select_best


class EvaluationTests(unittest.TestCase):
    def test_perfect(self):
        frame = pd.DataFrame({'y_true':['good','regular','bad'], 'y_pred':['good','regular','bad']})
        self.assertEqual(evaluate(frame), {'accuracy':1.0,'f1_macro':1.0})

    def test_minority_error(self):
        frame = pd.DataFrame({'y_true':['good','good','regular','bad'], 'y_pred':['good','good','good','bad']})
        result = evaluate(frame)
        self.assertAlmostEqual(result['accuracy'], 0.75)
        self.assertAlmostEqual(result['f1_macro'], 0.6)

    def test_tie_does_not_select_a_winner_or_include_the_dummy_model(self):
        scores = {'baseline': 1.0, 'logistic_regression': 0.6, 'svm': 0.8,
                  'random_forest': 0.7, 'cnn': 0.8 + 1e-15}
        selection = select_best(scores)
        self.assertEqual(selection['status'], 'tie')
        self.assertIsNone(selection['winner'])
        self.assertCountEqual(selection['tied_models'], ['svm', 'cnn'])

    def test_history_exports_vector_and_raster_curves(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            history = pd.DataFrame({'epoch': [1], 'accuracy': [0.8], 'val_accuracy': [0.7],
                                    'loss': [0.3], 'val_loss': [0.4]})
            history.to_csv(root / 'history.csv', index=False)
            plot_history(root / 'history.csv', root)
            for metric in ['loss', 'accuracy']:
                self.assertTrue((root / f'cnn_{metric}.png').is_file())
                self.assertIn('<svg', (root / f'cnn_{metric}.svg').read_text(encoding='utf-8'))

    def test_windows_predictions_match_manifest_and_incomplete_models_are_excluded(self):
        labels = ['good', 'regular', 'bad']
        # La selección usa validación y el reporte usa prueba: el manifiesto tiene ambos splits.
        manifest = pd.DataFrame({
            'path': [f'data/{label}.jpg' for label in labels] * 2,
            'split': ['val'] * 3 + ['test'] * 3, 'fruit': 'apple', 'quality': labels * 2,
        })
        predictions = pd.DataFrame({
            'image_path': manifest.path.str.replace('/', '\\', regex=False),
            'split': manifest.split, 'y_true': labels * 2, 'y_pred': labels * 2,
        })
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'evaluation'
            manifest.to_csv(root / 'splits.csv', index=False)
            predictions.to_csv(root / 'random_forest_preds.csv', index=False)
            for name in ['logistic_regression', 'svm', 'cnn']:
                predictions.assign(y_pred='good').to_csv(root / f'{name}_preds.csv', index=False)
            compare(root, root / 'splits.csv', output)
            selection_path = output / 'selection.json'
            selected = json.loads(selection_path.read_text(encoding='utf-8'))
            self.assertEqual(selected['status'], 'selected')
            self.assertEqual(selected['winner'], 'random_forest')
            self.assertEqual(selected['best_f1_macro'], 1.0)
            self.assertEqual(selected['split'], 'val')
            self.assertTrue((output / 'random_forest_test_confusion.svg').is_file())
            self.assertTrue((output / 'comparison.svg').is_file())
            (root / 'logistic_regression_preds.csv').unlink()
            (root / 'cnn_preds.csv').write_text('', encoding='utf-8')
            predictions.iloc[:2].to_csv(root / 'svm_preds.csv', index=False)
            compare(root, root / 'splits.csv', output)
            pending = json.loads(selection_path.read_text(encoding='utf-8'))
            self.assertEqual(pending['status'], 'pending')
            self.assertIsNone(pending['winner'])
            self.assertCountEqual(pending['missing_models'], ['logistic_regression', 'svm', 'cnn'])
            metrics = pd.read_csv(output / 'metrics.csv')
            self.assertEqual(metrics.to_dict('records'), [
                {'model': 'random_forest', 'split': 'val', 'n': 3, 'accuracy': 1.0, 'f1_macro': 1.0},
                {'model': 'random_forest', 'split': 'test', 'n': 3, 'accuracy': 1.0, 'f1_macro': 1.0},
            ])
            self.assertIn('svm/test: cobertura', (output / 'status.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
