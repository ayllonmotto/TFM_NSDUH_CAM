"""Prueba integral del entorno principal TFM_ML."""

from __future__ import annotations

import os
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import joblib
import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import pyarrow as pa
import pydot
import seaborn as sns
import shap
import tensorflow as tf
import xgboost as xgb

from imblearn.over_sampling import RandomOverSampler
from pypdf import PdfReader, PdfWriter
from ipywidgets import IntProgress
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
)
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


SEED = 12345

EXPECTED_VERSIONS = {
    "numpy": "2.4.6",
    "pandas": "3.0.5",
    "pyarrow": "25.0.0",
    "openpyxl": "3.1.5",
    "pypdf": "6.14.2",
    "pydot": "4.0.1",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.0",
    "imbalanced-learn": "0.14.2",
    "xgboost": "3.2.0",
    "shap": "0.51.0",
    "matplotlib": "3.11.1",
    "seaborn": "0.13.2",
    "joblib": "1.5.3",
    "ipykernel": "7.3.0",
    "ipywidgets": "8.1.8",
    "tensorflow": "2.20.0",
    "keras": "3.15.1",
    "tensorboard": "2.20.0",
}


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_versions() -> None:
    section("1. VERSIONES FIJADAS")

    errors: list[str] = []

    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "NO INSTALADO"

        state = "OK" if actual == expected else "DIFERENTE"

        print(
            f"{package:22s}: "
            f"{actual:14s} "
            f"esperada={expected:14s} "
            f"{state}"
        )

        if actual != expected:
            errors.append(
                f"{package}: instalada={actual}, esperada={expected}"
            )

    if errors:
        raise RuntimeError(
            "Versiones diferentes del entorno:\n- "
            + "\n- ".join(errors)
        )


def validate_gpu() -> None:
    section("2. TENSORFLOW Y GPU")

    print("TensorFlow:", tf.__version__)
    print("Compilado con CUDA:", tf.test.is_built_with_cuda())

    physical_gpus = tf.config.list_physical_devices("GPU")
    print("GPU físicas:", physical_gpus)

    if not tf.test.is_built_with_cuda():
        raise RuntimeError("TensorFlow no está compilado con CUDA.")

    if not physical_gpus:
        raise RuntimeError("TensorFlow no detecta ninguna GPU.")

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    logical_gpus = tf.config.list_logical_devices("GPU")
    print("GPU lógicas:", logical_gpus)

    tf.random.set_seed(SEED)

    with tf.device("/GPU:0"):
        matrix_a = tf.random.uniform((1000, 1000))
        matrix_b = tf.random.uniform((1000, 1000))
        matrix_c = tf.matmul(matrix_a, matrix_b)
        matrix_mean = tf.reduce_mean(matrix_c)

    print("Dispositivo:", matrix_c.device)
    print("Forma:", matrix_c.shape)
    print("Media:", float(matrix_mean.numpy()))

    if "GPU:0" not in matrix_c.device.upper():
        raise RuntimeError(
            "La operación matricial no se ejecutó en la GPU."
        )


def validate_scientific_stack(work_dir: Path) -> None:
    section("3. PILA CIENTÍFICA, GRÁFICOS Y WIDGETS")

    matrix = np.arange(20, dtype=float).reshape(5, 4)
    frame = pd.DataFrame(matrix, columns=list("ABCD"))

    print("Forma NumPy:", matrix.shape)
    print("Forma pandas:", frame.shape)
    print("Media columna C:", float(frame["C"].mean()))

    figure, axis = plt.subplots(figsize=(4, 3))
    sns.scatterplot(x=frame["A"], y=frame["B"], ax=axis)
    plt.close(figure)

    progress = IntProgress(value=2, min=0, max=10)

    print("Matplotlib/Seaborn: OK")
    print("Widget:", type(progress).__name__)
    print("Valor del widget:", progress.value)

    print("PyArrow importado:", pa.__version__)
    print("openpyxl importado:", openpyxl.__version__)

    parquet_path = work_dir / "prueba_pyarrow.parquet"
    frame.to_parquet(
        parquet_path,
        engine="pyarrow",
        index=False,
    )
    parquet_restored = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
    )

    if not frame.equals(parquet_restored):
        raise RuntimeError(
            "La escritura/lectura Parquet con PyArrow no conservó los datos."
        )

    print("Parquet con PyArrow: OK")

    excel_path = work_dir / "prueba_openpyxl.xlsx"
    frame.to_excel(
        excel_path,
        engine="openpyxl",
        index=False,
    )
    excel_restored = pd.read_excel(
        excel_path,
        engine="openpyxl",
    )

    if (
        excel_restored.shape != frame.shape
        or not np.allclose(
            excel_restored.to_numpy(dtype=float),
            frame.to_numpy(dtype=float),
        )
    ):
        raise RuntimeError(
            "La escritura/lectura Excel con openpyxl no conservó los datos."
        )

    print("Excel con openpyxl: OK")

    pdf_path = work_dir / "prueba_pypdf.pdf"
    writer = PdfWriter()
    writer.add_blank_page(
        width=72,
        height=72,
    )

    with pdf_path.open("wb") as file:
        writer.write(file)

    reader = PdfReader(str(pdf_path))

    if len(reader.pages) != 1:
        raise RuntimeError(
            "La creación/lectura del PDF con pypdf no es correcta."
        )

    print("PDF con pypdf: OK")


def validate_graphviz(work_dir: Path) -> None:
    section("4. PYDOT, GRAPHVIZ Y VISUALIZACIÓN KERAS")

    dot_path = shutil.which("dot")

    print("pydot:", pydot.__version__)
    print("Graphviz dot:", dot_path)

    if dot_path is None:
        raise RuntimeError(
            "No se encontró el ejecutable 'dot' de Graphviz."
        )

    graph_path = work_dir / "prueba_pydot.png"

    graph = pydot.Dot(
        graph_type="digraph",
        rankdir="LR",
    )
    graph.add_node(pydot.Node("entrada"))
    graph.add_node(pydot.Node("modelo"))
    graph.add_node(pydot.Node("salida"))
    graph.add_edge(pydot.Edge("entrada", "modelo"))
    graph.add_edge(pydot.Edge("modelo", "salida"))

    graph.write_png(str(graph_path))

    if not graph_path.exists() or graph_path.stat().st_size == 0:
        raise RuntimeError(
            "pydot y Graphviz no generaron correctamente el PNG."
        )

    print(
        "pydot + Graphviz: OK",
        f"({graph_path.stat().st_size} bytes)",
    )

    keras_path = work_dir / "prueba_keras_plot_model.png"

    inputs = keras.Input(
        shape=(10,),
        name="entrada_grafico",
    )
    hidden = keras.layers.Dense(
        8,
        activation="relu",
        name="densa_grafico",
    )(inputs)
    outputs = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="salidas_grafico",
    )(hidden)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="prueba_grafico_tfm",
    )

    keras.utils.plot_model(
        model,
        to_file=str(keras_path),
        show_shapes=True,
        show_layer_names=True,
    )

    if not keras_path.exists() or keras_path.stat().st_size == 0:
        raise RuntimeError(
            "keras.utils.plot_model() no generó correctamente el PNG."
        )

    print(
        "keras.utils.plot_model: OK",
        f"({keras_path.stat().st_size} bytes)",
    )


def validate_machine_learning(work_dir: Path) -> None:
    section("5. MACHINE LEARNING MULTISALIDA")

    X, y = make_multilabel_classification(
        n_samples=240,
        n_features=14,
        n_classes=4,
        n_labels=2,
        random_state=SEED,
    )

    model = MultiOutputClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=SEED,
        )
    )

    model.fit(X, y)
    predictions = model.predict(X[:12])

    print("Forma de los objetivos:", y.shape)
    print("Forma de las predicciones:", predictions.shape)

    if predictions.shape != (12, 4):
        raise RuntimeError(
            "La clasificación multisalida tiene una forma incorrecta."
        )

    model_path = work_dir / "modelo_ml.joblib"

    joblib.dump(model, model_path)
    restored_model = joblib.load(model_path)
    restored_predictions = restored_model.predict(X[:12])

    if not np.array_equal(predictions, restored_predictions):
        raise RuntimeError(
            "Las predicciones cambiaron tras restaurar el modelo."
        )

    print("Persistencia con Joblib: OK")


def validate_imbalance_xgboost_shap() -> None:
    section("6. DESBALANCEO, XGBOOST Y SHAP")

    X, y = make_classification(
        n_samples=320,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        weights=[0.88, 0.12],
        random_state=SEED,
    )

    before = np.bincount(y)

    sampler = RandomOverSampler(random_state=SEED)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    after = np.bincount(y_resampled)

    print("Distribución original:", before.tolist())
    print("Distribución remuestreada:", after.tolist())

    if after[0] != after[1]:
        raise RuntimeError(
            "RandomOverSampler no equilibró las clases."
        )

    model = xgb.XGBClassifier(
        n_estimators=12,
        max_depth=3,
        learning_rate=0.10,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=2,
    )

    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X[:6])

    print("Forma SHAP:", explanation.values.shape)
    print(
        "Valores SHAP finitos:",
        bool(np.isfinite(explanation.values).all()),
    )

    if explanation.values.shape != (6, 12):
        raise RuntimeError("La forma de SHAP no es la esperada.")

    if not np.isfinite(explanation.values).all():
        raise RuntimeError("SHAP produjo valores no finitos.")


def validate_deep_learning(work_dir: Path) -> None:
    section("7. DEEP LEARNING MULTISALIDA")

    rng = np.random.default_rng(SEED)

    X = rng.normal(size=(256, 12)).astype(np.float32)

    targets = {
        "episodio_depresivo": rng.integers(
            0, 2, size=(256, 1)
        ).astype(np.float32),
        "ideacion_suicida": rng.integers(
            0, 2, size=(256, 1)
        ).astype(np.float32),
        "planificacion_suicida": rng.integers(
            0, 2, size=(256, 1)
        ).astype(np.float32),
        "intento_suicida": rng.integers(
            0, 2, size=(256, 1)
        ).astype(np.float32),
    }

    normalization = keras.layers.Normalization(
        name="normalizacion"
    )
    normalization.adapt(X)

    inputs = keras.Input(
        shape=(12,),
        name="variables_predictoras",
    )

    shared = normalization(inputs)
    shared = keras.layers.Dense(
        24,
        activation="relu",
        name="capa_compartida",
    )(shared)

    outputs = {
        name: keras.layers.Dense(
            1,
            activation="sigmoid",
            name=name,
        )(shared)
        for name in targets
    }

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="smoke_test_multisalida",
    )

    model.compile(
        optimizer="adam",
        loss={
            name: "binary_crossentropy"
            for name in targets
        },
    )

    model.fit(
        X,
        targets,
        epochs=1,
        batch_size=32,
        verbose=0,
    )

    with tf.device("/GPU:0"):
        prediction_tensors = model(
            tf.convert_to_tensor(X[:8]),
            training=False,
        )

    for name, tensor in prediction_tensors.items():
        values = tensor.numpy()

        print(
            f"{name:25s} "
            f"forma={values.shape} "
            f"dispositivo={tensor.device}"
        )

        if values.shape != (8, 1):
            raise RuntimeError(
                f"Forma incorrecta en la salida {name}."
            )

        if "GPU:0" not in tensor.device.upper():
            raise RuntimeError(
                f"La salida {name} no se calculó en la GPU."
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"La salida {name} contiene valores no finitos."
            )

    model_path = work_dir / "modelo_dl.keras"
    model.save(model_path)

    restored_model = keras.models.load_model(model_path)
    restored_predictions = restored_model.predict(
        X[:8],
        verbose=0,
    )

    if not isinstance(restored_predictions, dict):
        raise RuntimeError(
            "El modelo restaurado no devolvió un diccionario."
        )

    if set(restored_predictions) != set(targets):
        raise RuntimeError(
            "Las salidas cambiaron al restaurar el modelo."
        )

    for name, values in restored_predictions.items():
        if values.shape != (8, 1):
            raise RuntimeError(
                f"Forma incorrecta tras restaurar {name}."
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Valores no finitos tras restaurar {name}."
            )

    print("Guardado y restauración Keras: OK")


def main() -> None:
    work_dir = Path(
        tempfile.mkdtemp(prefix="tfm_ml_smoke_test_")
    )

    try:
        validate_versions()
        validate_gpu()
        validate_scientific_stack(work_dir)
        validate_graphviz(work_dir)
        validate_machine_learning(work_dir)
        validate_imbalance_xgboost_shap()
        validate_deep_learning(work_dir)

        section("ENTORNO TFM_ML VALIDADO COMPLETAMENTE")
        print("Todas las pruebas finalizaron correctamente.")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
