"""Entrenamiento de la CNN. Andrés Fajardo.

Ejecutar desde la raíz:
    python -m src.training.train_cnn                          # entrena todo de una vez
    python -m src.training.train_cnn --max-seconds 150        # por partes (reanudable)

Reanudable: guarda el estado al final de cada época (experiments/checkpoints/), así que si
se interrumpe continúa desde la última época completa. Al terminar escribe
experiments/results/cnn_preds.csv y las curvas de aprendizaje.

Pesos de clase (--class-weight):
    full : w_c = N / (K · N_c)            → regular (≈1 % de train) recibe un peso ≈ 40
    sqrt : w_c = sqrt(N / (K · N_c))      → por defecto; suaviza el peso (≈ 6) y evita que la
           red colapse y deje de predecir alguna clase.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import QUALITY_TO_ID, ID_TO_QUALITY, load_splits
from src.models.cnn import build_cnn

RESULTS = Path('experiments/results')
CHECKPOINTS = Path('experiments/checkpoints')
LOGS = Path('experiments/logs')
STATE = CHECKPOINTS / 'cnn_state.json'
LAST = CHECKPOINTS / 'cnn_last.keras'
BEST = CHECKPOINTS / 'cnn.keras'
HISTORY = LOGS / 'cnn_history.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--class-weight', choices=['full', 'sqrt'], default='sqrt')
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--restart', action='store_true', help='ignorar el estado guardado y empezar de cero')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs y batch-size deben ser positivos')
    t0 = time.time()

    df = load_splits()
    for split in ('train', 'val', 'test'):
        part = df[df.split == split]
        if set(part.quality) != set(QUALITY_TO_ID):
            raise SystemExit(f'{split}: faltan clases o hay etiquetas inválidas. Actualizar splits antes del entrenamiento final.')
    missing = [p for p in df.processed_path if not Path(p).is_file()]
    stats_path = Path('data/processed/stats.json')
    if missing or not stats_path.is_file():
        raise SystemExit(f'Faltan {len(missing)} imágenes procesadas o stats.json. Ejecutar src.data.preprocess.')

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    stats = json.loads(stats_path.read_text())
    mean = tf.constant(stats['mean_rgb'], dtype=tf.float32)
    std = tf.constant(stats['std_rgb'], dtype=tf.float32)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=42),
        tf.keras.layers.RandomRotation(0.5, fill_mode='constant', seed=42),
    ])

    def dataset(split, training=False):
        meta = df[df.split == split].reset_index(drop=True)
        labels = meta.quality.map(QUALITY_TO_ID).to_numpy(dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((meta.processed_path.astype(str).tolist(), labels))
        if training:
            ds = ds.shuffle(len(meta), seed=42, reshuffle_each_iteration=True)

        def prepare(path, label):
            img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.ensure_shape(img, (128, 128, 3))
            img = tf.cast(img, tf.float32) / 255.0
            if training:
                img = augmentation(img, training=True)
            return (img - mean) / std, label
        return ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE), meta

    for d in (RESULTS, CHECKPOINTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    # El estado guarda la huella del split: si splits.csv cambió, se empieza de cero.
    fingerprint = str(pd.util.hash_pandas_object(df[['path', 'quality', 'split']], index=False).sum())
    state = json.loads(STATE.read_text()) if STATE.exists() and not args.restart else {}
    if state.get('fingerprint') != fingerprint:
        state = {'fingerprint': fingerprint, 'epoch': 0, 'best_val_loss': None, 'best_epoch': None,
                 'wait': 0, 'done': False, 'class_weight': args.class_weight}
        if HISTORY.exists():
            HISTORY.unlink()

    train, train_meta = dataset('train', True)
    val, _ = dataset('val')
    counts = train_meta.quality.value_counts()
    raw = {i: len(train_meta) / (3 * counts[label]) for label, i in QUALITY_TO_ID.items()}
    weights = {i: (float(np.sqrt(w)) if state['class_weight'] == 'sqrt' else float(w)) for i, w in raw.items()}
    print('Pesos de clase:', {ID_TO_QUALITY[i]: round(w, 2) for i, w in weights.items()})

    model = tf.keras.models.load_model(LAST) if state['epoch'] > 0 and LAST.exists() else build_cnn()
    if state['epoch'] == 0:
        model.summary()

    epoch_time = None
    while not state['done'] and state['epoch'] < args.epochs:
        if args.max_seconds and epoch_time and time.time() - t0 + epoch_time > args.max_seconds:
            print(f"Pausa: {state['epoch']} épocas completas. Vuelve a correr el comando para continuar.")
            return
        e0 = time.time()
        h = model.fit(train, validation_data=val, epochs=state['epoch'] + 1, initial_epoch=state['epoch'],
                      class_weight=weights, verbose=2).history
        epoch_time = time.time() - e0
        state['epoch'] += 1
        row = {'epoch': state['epoch'], **{k: v[-1] for k, v in h.items()}}
        pd.DataFrame([row]).to_csv(HISTORY, mode='a', header=not HISTORY.exists(), index=False)
        model.save(LAST)
        vl = row['val_loss']
        if state['best_val_loss'] is None or vl < state['best_val_loss']:
            state.update(best_val_loss=vl, best_epoch=state['epoch'], wait=0)
            model.save(BEST)
        else:
            state['wait'] += 1
            if state['wait'] >= args.patience:
                state['done'] = True       # early stopping
        STATE.write_text(json.dumps(state, indent=2))
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"Pausa: {state['epoch']} épocas completas. Vuelve a correr el comando para continuar.")
            return
    state['done'] = True
    STATE.write_text(json.dumps(state, indent=2))
    print(f"Entrenamiento terminado. Mejor época: {state['best_epoch']} (val_loss={state['best_val_loss']:.4f})")

    model = tf.keras.models.load_model(BEST)
    frames = []
    for split in ('train', 'val', 'test'):
        ds, meta = dataset(split)
        probs = model.predict(ds, verbose=0)
        pred = pd.DataFrame({'image_path': meta.path, 'y_true': meta.quality,
                             'y_pred': [ID_TO_QUALITY[i] for i in probs.argmax(axis=1)], 'split': split})
        for label, i in QUALITY_TO_ID.items():
            pred['prob_' + label] = probs[:, i]
        frames.append(pred)
    pd.concat(frames, ignore_index=True).to_csv(RESULTS / 'cnn_preds.csv', index=False)
    from src.evaluation.compare_models import plot_history
    plot_history(HISTORY, RESULTS / 'evaluation')
    print('Predicciones guardadas en experiments/results/cnn_preds.csv')


if __name__ == '__main__':
    main()
