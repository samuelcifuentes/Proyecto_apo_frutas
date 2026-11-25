"""Entrenamiento de la CNN. Andrés Fajardo.
Ejecutar desde la raíz: python -m src.training.train_cnn
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.cnn import build_cnn
from src.data.dataset import QUALITY_TO_ID, ID_TO_QUALITY, load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs y batch-size deben ser positivos')
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

    train, train_meta = dataset('train', True)
    val, _ = dataset('val')
    results = Path('experiments/results'); results.mkdir(parents=True, exist_ok=True)
    checkpoints = Path('experiments/checkpoints'); checkpoints.mkdir(parents=True, exist_ok=True)
    logs = Path('experiments/logs'); logs.mkdir(parents=True, exist_ok=True)
    counts = train_meta.quality.value_counts()
    weights = {i: len(train_meta)/(3*counts[label]) for label, i in QUALITY_TO_ID.items()}
    model = build_cnn()
    model.summary()
    history = model.fit(train, validation_data=val, epochs=args.epochs, class_weight=weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                   tf.keras.callbacks.ModelCheckpoint(str(checkpoints/'cnn.keras'), monitor='val_loss', save_best_only=True)])
    pd.DataFrame(history.history).rename_axis('epoch').reset_index().assign(epoch=lambda x: x.epoch+1).to_csv(logs/'cnn_history.csv', index=False)
    model = tf.keras.models.load_model(checkpoints/'cnn.keras')
    frames = []
    for split in ('train', 'val', 'test'):
        ds, meta = dataset(split)
        probs = model.predict(ds, verbose=0)
        pred = pd.DataFrame({'image_path': meta.path, 'y_true': meta.quality,
                             'y_pred': [ID_TO_QUALITY[i] for i in probs.argmax(axis=1)], 'split': split})
        for label, i in QUALITY_TO_ID.items():
            pred['prob_'+label] = probs[:, i]
        frames.append(pred)
    pd.concat(frames, ignore_index=True).to_csv(results/'cnn_preds.csv', index=False)
    from src.evaluation.compare_models import plot_history
    plot_history(logs/'cnn_history.csv', results/'evaluation')


if __name__ == '__main__':
    main()
