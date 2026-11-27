"""Comparación reproducible de modelos. Andrés Fajardo."""
from pathlib import Path
import argparse
import json
from math import isclose
from tempfile import TemporaryDirectory
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

LABELS = ['good', 'regular', 'bad']
CANDIDATES = ['logistic_regression', 'svm', 'random_forest', 'cnn']
MODELS = ['baseline', *CANDIDATES]


def save_figure(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f'{name}.png', dpi=160)
    fig.savefig(output / f'{name}.svg')
    plt.close(fig)


def plot_history(history_path, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history_path)
    for metric in ['loss', 'accuracy']:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history.epoch, history[metric], label='Entrenamiento')
        ax.plot(history.epoch, history['val_'+metric], label='Validación')
        ax.set(xlabel='Época', ylabel=metric, title='CNN: '+metric)
        ax.legend()
        save_figure(fig, output, f'cnn_{metric}')


def evaluate(frame):
    return {'accuracy': accuracy_score(frame.y_true, frame.y_pred),
            'f1_macro': f1_score(frame.y_true, frame.y_pred, labels=LABELS, average='macro', zero_division=0)}


def select_best(test_scores, split='val'):
    """Selecciona entre los cuatro candidatos por F1-macro del split indicado.

    La selección se hace con VALIDACIÓN: el conjunto de prueba solo se usa para reportar
    el desempeño final del modelo ya elegido. Elegir con test haría que la cifra de test
    dejara de ser una estimación imparcial del desempeño en frutas nuevas.
    """
    missing = [name for name in CANDIDATES if name not in test_scores]
    selection = {'split': split, 'metric': 'f1_macro', 'status': 'pending',
                 'winner': None, 'best_f1_macro': None, 'missing_models': missing,
                 'tied_models': []}
    if missing:
        return selection
    best = max(test_scores[name] for name in CANDIDATES)
    leaders = [name for name in CANDIDATES
               if isclose(test_scores[name], best, rel_tol=0.0, abs_tol=1e-12)]
    selection['best_f1_macro'] = best
    if len(leaders) == 1:
        selection.update(status='selected', winner=leaders[0])
    else:
        selection.update(status='tie', tied_models=leaders)
    return selection


def compare(results, manifest, output):
    output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(manifest).rename(columns={'path':'image_path', 'quality':'expected'})
    # El extractor clásico usa rutas nativas y el manifiesto usa barras de POSIX.
    meta['image_path'] = meta['image_path'].str.replace('\\', '/', regex=False)
    if meta.duplicated(['image_path','split']).any():
        raise ValueError('El manifiesto contiene rutas duplicadas dentro de un split')
    rows, notes, gaps = [], [], []
    for name in MODELS:
        path = results/f'{name}_preds.csv'
        if not path.exists():
            notes.append(f'{name}: predicciones pendientes.'); continue
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            notes.append(f'{name}: archivo CSV ilegible.'); continue
        required = {'image_path','split','y_true','y_pred'}
        if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
            notes.append(f'{name}: formato inválido o valores faltantes.'); continue
        frame['image_path'] = frame['image_path'].astype(str).str.replace('\\', '/', regex=False)
        if frame.duplicated(['image_path','split']).any() or not set(frame.y_pred).issubset(LABELS):
            notes.append(f'{name}: rutas duplicadas o predicciones inválidas.'); continue
        # Solo se comparan resultados con cobertura completa del mismo split.
        scores = {}
        for split in ['train','val','test']:
            expected = meta[meta.split == split]
            part = frame[frame.split == split]
            joined = expected.merge(part, on=['image_path','split'], how='outer', indicator=True)
            if joined.empty or not joined['_merge'].eq('both').all() or not joined.expected.eq(joined.y_true).all():
                notes.append(f'{name}/{split}: cobertura o etiquetas distintas del manifiesto; no comparable.'); continue
            if set(joined.y_true) != set(LABELS):
                notes.append(f'{name}/{split}: faltan clases reales; evaluación de tres clases no válida.'); continue
            metrics = evaluate(joined); scores[split] = metrics
            rows.append({'model':name,'split':split,'n':len(joined),**metrics})
            matrix = confusion_matrix(joined.y_true, joined.y_pred, labels=LABELS)
            fig, ax = plt.subplots(figsize=(5,4))
            ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'{name} - {split}')
            save_figure(fig, output, f'{name}_{split}_confusion')
            pd.DataFrame(classification_report(joined.y_true, joined.y_pred, labels=LABELS, output_dict=True, zero_division=0)).T.to_csv(output/f'{name}_{split}_report.csv')
            if split == 'test':
                errors = joined[joined.y_true != joined.y_pred].drop(columns='_merge')
                errors.to_csv(output/f'{name}_errors.csv', index=False)
                errors.groupby(['fruit','y_true','y_pred']).size().rename('count').reset_index().to_csv(output/f'{name}_error_groups.csv', index=False)
        if 'train' in scores and 'val' in scores:
            gaps.append({'model':name, **{k+'_gap':scores['train'][k]-scores['val'][k] for k in scores['train']}})
    metrics = pd.DataFrame(rows, columns=['model','split','n','accuracy','f1_macro'])
    metrics.to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(gaps, columns=['model','accuracy_gap','f1_macro_gap']).to_csv(output/'overfitting.csv', index=False)
    test = metrics[metrics.split == 'test'].sort_values('f1_macro', ascending=False)
    val = metrics[metrics.split == 'val']
    selection = select_best(dict(zip(val.model, val.f1_macro)), split='val')
    # F1-macro de test de cada candidato, solo para reportar (no interviene en la elección)
    selection['test_f1_macro'] = {m: round(float(f), 4) for m, f in zip(test.model, test.f1_macro)
                                  if m in CANDIDATES}
    with TemporaryDirectory(dir=output) as staging:
        selection_file = Path(staging) / 'selection.json'
        selection_file.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
        selection_file.replace(output / 'selection.json')
    if not test.empty:
        ax = test.set_index('model')[['accuracy','f1_macro']].plot.bar(rot=20, ylim=(0,1), figsize=(8,4))
        ax.set_ylabel('Puntuación')
        save_figure(ax.figure, output, 'comparison')
    (output/'status.txt').write_text('\n'.join(notes) or 'Evaluación completa.', encoding='utf-8')
    print(metrics.to_string(index=False) if not metrics.empty else 'No hay resultados comparables todavía.')
    print('\n'.join(notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('experiments/results'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits.csv'))
    args = parser.parse_args()
    output = args.results/'evaluation'
    compare(args.results, args.manifest, output)
    history = Path('experiments/logs/cnn_history.csv')
    if history.exists():
        plot_history(history, output)


if __name__ == '__main__':
    main()
