"""Entrenamiento y evaluación de modelos clásicos para calidad de frutas."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import FEATURE_NAMES

LOGGER = logging.getLogger(__name__)
QUALITY_LABELS = ("good", "regular", "bad")
MODEL_NAMES = ("baseline", "svm", "random_forest", "logistic_regression")
METADATA_COLUMNS = {
    "annotator",
    "d_norm",
    "group",
    "filename",
    "fruit",
    "image_path",
    "path",
    "processed_path",
    "quality",
    "segmentation_ok",
    "size",
    "source",
    "split",
    "y_pred",
    "y_true",
}


def build_grid_search(model_name: str) -> GridSearchCV:
    """Construye el Pipeline y el GridSearchCV estratificado de un modelo."""
    if model_name == "baseline":
        pipeline = Pipeline(
            [
                (
                    "classifier",
                    DummyClassifier(strategy="most_frequent", random_state=42),
                )
            ]
        )
        parameter_grid: dict[str, list[Any]] = {}
    elif model_name == "svm":
        svm_calibration_cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=42
        )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=SVC(class_weight="balanced", random_state=42),
                        cv=svm_calibration_cv,
                        ensemble=False,
                    ),
                ),
            ]
        )
        parameter_grid = {
            "classifier__estimator__C": [0.1, 1.0, 10.0],
            "classifier__estimator__kernel": ["linear", "rbf"],
            "classifier__estimator__gamma": ["scale", "auto"],
        }
    elif model_name == "random_forest":
        pipeline = Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced", random_state=42, n_jobs=1
                    ),
                )
            ]
        )
        parameter_grid = {
            "classifier__n_estimators": [100, 250],
            "classifier__max_depth": [None, 15],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt"],
        }
    elif model_name == "logistic_regression":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=42
                    ),
                ),
            ]
        )
        parameter_grid = {"classifier__C": [0.1, 1.0, 10.0]}
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        return_train_score=True,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )


def _normalise_label(label: object) -> str:
    """Convierte códigos numéricos de calidad a los nombres compartidos del proyecto."""
    if pd.isna(label):
        return ""
    if isinstance(label, (int, np.integer, float, np.floating)):
        if np.isfinite(label) and float(label).is_integer():
            return {0: "good", 1: "regular", 2: "bad"}.get(int(label), str(int(label)))
    return str(label).strip().lower()


def _select_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Selecciona características conocidas o columnas numéricas no-metadato."""
    known = [name for name in FEATURE_NAMES if name in dataframe.columns]
    if known:
        return known
    return [
        column
        for column in dataframe.select_dtypes(include=[np.number]).columns
        if column not in METADATA_COLUMNS
    ]


def _read_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Carga divisiones desde un CSV combinado o desde archivos separados."""
    if args.features:
        dataframe = pd.read_csv(args.features)
        if "split" not in dataframe.columns:
            raise ValueError("El CSV combinado debe incluir una columna 'split'")
        return {
            split: dataframe.loc[dataframe["split"].astype(str).str.lower() == split]
            .copy()
            .reset_index(drop=True)
            for split in ("train", "val", "test")
        }

    if not args.train:
        raise ValueError("Se debe indicar --features o --train")
    splits = {"train": pd.read_csv(args.train)}
    for split, filename in (("val", args.val), ("test", args.test)):
        if filename:
            splits[split] = pd.read_csv(filename)
    return splits


def _prepare_split(
    dataframe: pd.DataFrame, split_name: str, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Valida las etiquetas y elimina filas sin características numéricas válidas."""
    label_column = "quality" if "quality" in dataframe.columns else "y_true"
    if label_column not in dataframe.columns:
        raise ValueError(f"El CSV de {split_name} debe incluir 'quality' o 'y_true'")
    if not feature_columns:
        raise ValueError("No se encontraron columnas numéricas de características")

    features = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")
    labels = dataframe[label_column].map(_normalise_label)
    valid = (
        labels.notna()
        & labels.ne("")
        & labels.ne("discard")
        & np.isfinite(features).all(axis=1)
    )
    dropped = int((~valid).sum())
    if dropped:
        LOGGER.warning(
            "Se omiten %d filas sin etiqueta válida o con características faltantes en %s",
            dropped,
            split_name,
        )

    valid_data = dataframe.loc[valid]
    image_column = next(
        (name for name in ("image_path", "path", "filename") if name in valid_data),
        None,
    )
    image_paths = (
        valid_data[image_column].fillna("").astype(str).tolist()
        if image_column
        else [""] * int(valid.sum())
    )
    return (
        features.loc[valid].to_numpy(dtype=np.float64),
        labels.loc[valid].to_numpy(dtype=str),
        image_paths,
    )


def _ordered_labels(labels: np.ndarray) -> list[str]:
    """Devuelve primero las etiquetas del proyecto y luego etiquetas adicionales."""
    present = set(labels.astype(str))
    return [label for label in QUALITY_LABELS if label in present] + sorted(
        present.difference(QUALITY_LABELS)
    )


def _print_metrics(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> None:
    """Imprime accuracy, F1-macro y matriz de confusión para una división."""
    print(f"\n{model_name} — {split_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("Matriz de confusión (filas=real, columnas=predicción)")
    print(f"Etiquetas: {labels}")
    print(confusion_matrix(y_true, y_pred, labels=labels))


def _save_confusion_figure(
    matrix: np.ndarray, labels: list[str], output_path: Path, title: str
) -> None:
    """Guarda una matriz de confusión como gráfica SVG vectorial."""
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicción",
        ylabel="Valor real",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _probability_column(label: str) -> str:
    """Crea un nombre de columna estable para la probabilidad de una clase."""
    safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_")
    return f"prob_{safe_label}"


def _prediction_rows(
    estimator: Pipeline,
    x_values: np.ndarray,
    y_true: np.ndarray,
    image_paths: list[str],
    split_name: str,
) -> pd.DataFrame:
    """Crea filas de predicción, incluyendo probabilidades cuando están disponibles."""
    y_pred = estimator.predict(x_values).astype(str)
    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "y_true": y_true.astype(str),
            "y_pred": y_pred,
            "split": split_name,
        }
    )
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_values)
        classifier = estimator.named_steps["classifier"]
        for index, label in enumerate(classifier.classes_):
            predictions[_probability_column(str(label))] = probabilities[:, index]
    return predictions


def _save_feature_importance(
    estimator: Pipeline, feature_columns: list[str], output_path: Path
) -> None:
    """Guarda las importancias del bosque aleatorio en formato vectorial SVG."""
    classifier = estimator.named_steps["classifier"]
    importances = classifier.feature_importances_
    order = np.argsort(importances)
    figure, axis = plt.subplots(figsize=(9, max(5, 0.32 * len(feature_columns))))
    axis.barh(np.asarray(feature_columns)[order], importances[order], color="#4C78A8")
    axis.set(xlabel="Importancia", title="Importancia de características — Random Forest")
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)


def train_model(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    model_dir: Path,
    results_dir: Path,
) -> None:
    """Ajusta, evalúa y persiste un modelo y sus resultados."""
    prepared = {}
    for name, dataframe in splits.items():
        if dataframe.empty:
            continue
        split_data = _prepare_split(dataframe, name, feature_columns)
        if split_data[1].size:
            prepared[name] = split_data
        else:
            LOGGER.warning("No hay filas válidas para evaluar en %s", name)
    if "train" not in prepared:
        raise ValueError("No hay datos válidos en el split train")
    x_train, y_train, _ = prepared["train"]
    classes, counts = np.unique(y_train, return_counts=True)
    distribution = dict(zip(classes.tolist(), counts.tolist()))
    print(f"\nDistribución de clases en train: {distribution}")
    if len(classes) < 2:
        raise ValueError("El entrenamiento requiere al menos dos clases")
    if np.min(counts) < 5:
        raise ValueError(
            "StratifiedKFold(5) requiere al menos 5 ejemplos por clase en train; "
            f"distribución actual: {distribution}"
        )

    search = build_grid_search(model_name)
    search.fit(x_train, y_train)
    best_index = search.best_index_
    mean_train_score = float(search.cv_results_["mean_train_score"][best_index])
    mean_validation_score = float(search.cv_results_["mean_test_score"][best_index])
    print(f"{model_name} mejores hiperparámetros: {search.best_params_}")
    print(f"{model_name} F1-macro CV train: {mean_train_score:.4f}")
    print(f"{model_name} F1-macro CV validación: {mean_validation_score:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, model_dir / f"{model_name}.joblib")
    params = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_f1_macro": mean_validation_score,
        "mean_train_f1_macro": mean_train_score,
        "class_distribution_train": distribution,
        "feature_columns": feature_columns,
    }
    (model_dir / f"{model_name}_best_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(search.cv_results_).to_csv(
        results_dir / f"{model_name}_cv_results.csv", index=False
    )

    estimator = search.best_estimator_
    prediction_frames = []
    labels = _ordered_labels(y_train)
    for split_name in ("train", "val", "test"):
        if split_name not in prepared:
            continue
        x_values, y_true, image_paths = prepared[split_name]
        y_pred = estimator.predict(x_values).astype(str)
        _print_metrics(model_name, split_name, y_true, y_pred, labels)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        _save_confusion_figure(
            matrix,
            labels,
            figures_dir / f"{model_name}_confusion_{split_name}.svg",
            f"{model_name} — {split_name}",
        )
        prediction_frames.append(
            _prediction_rows(estimator, x_values, y_true, image_paths, split_name)
        )

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        results_dir / f"{model_name}_preds.csv", index=False
    )
    if model_name == "random_forest":
        _save_feature_importance(
            estimator,
            feature_columns,
            figures_dir / "random_forest_feature_importance.svg",
        )


def parse_args() -> argparse.Namespace:
    """Define los argumentos del comando de entrenamiento."""
    parser = argparse.ArgumentParser(
        description="Entrena modelos clásicos con características ya extraídas."
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="CSV combinado con columnas de características, quality y split.",
    )
    parser.add_argument("--train", type=Path, help="CSV de características de train.")
    parser.add_argument("--val", type=Path, help="CSV de características de val.")
    parser.add_argument("--test", type=Path, help="CSV de características de test.")
    parser.add_argument(
        "--model",
        choices=(*MODEL_NAMES, "all"),
        default="all",
        help="Modelo a entrenar (por defecto, todos).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("src/models/saved"),
        help="Directorio de modelos joblib y mejores parámetros JSON.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directorio para predicciones, CV y figuras.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta el entrenamiento solicitado desde la línea de comandos."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.features and any((args.train, args.val, args.test)):
        raise ValueError("--features no se combina con --train, --val o --test")

    splits = _read_splits(args)
    train_dataframe = splits.get("train", pd.DataFrame())
    if train_dataframe.empty:
        raise ValueError("El CSV de train está vacío o no fue proporcionado")
    feature_columns = _select_feature_columns(train_dataframe)
    if not feature_columns:
        raise ValueError("No se encontraron columnas de características numéricas")

    models = MODEL_NAMES if args.model == "all" else (args.model,)
    for model_name in models:
        train_model(
            model_name,
            splits,
            feature_columns,
            args.model_dir,
            args.results_dir,
        )


if __name__ == "__main__":
    main()
