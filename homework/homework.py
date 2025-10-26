# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


class Paths:
    BASE = Path("files")
    INPUT = BASE / "input"
    MODELS = BASE / "models"
    OUTPUT = BASE / "output"

    TRAIN = INPUT / "train_data.csv.zip"
    TEST = INPUT / "test_data.csv.zip"
    MODEL_FILE = MODELS / "model.pkl.gz"
    METRICS_FILE = OUTPUT / "metrics.json"


# ---------------------------------------------------------------------------
# Limpieza de datos
# ---------------------------------------------------------------------------

def preparar_datos(archivo_zip: Path) -> pd.DataFrame:
    """Lee, limpia y prepara los datos según las reglas especificadas."""
    data = pd.read_csv(archivo_zip, compression="zip").copy()

    # Renombrar y eliminar columnas no necesarias
    if "default payment next month" in data.columns:
        data.rename(columns={"default payment next month": "default"}, inplace=True)
    data.drop(columns=[c for c in ["ID"] if c in data.columns], inplace=True, errors="ignore")

    # Filtrar valores inválidos
    data = data[(data["MARRIAGE"] != 0) & (data["EDUCATION"] != 0)]
    data["EDUCATION"] = data["EDUCATION"].apply(lambda v: 4 if v > 4 else v)

    return data.dropna()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def limpiar_carpeta(path: Path) -> None:
    """Elimina y recrea el contenido de una carpeta."""
    if path.exists():
        for archivo in glob(str(path / "*")):
            os.remove(archivo)
    path.mkdir(parents=True, exist_ok=True)


def guardar_comprimido(obj: Any, ruta: Path) -> None:
    """Guarda un objeto serializado en formato gzip."""
    with gzip.open(ruta, "wb") as f:
        pickle.dump(obj, f)


def registrar_metricas(lista_registros: List[Dict[str, Any]], destino: Path) -> None:
    """Guarda métricas o matrices de confusión en formato JSON línea a línea."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        for registro in lista_registros:
            f.write(json.dumps(registro) + "\n")


# ---------------------------------------------------------------------------
# Métricas y evaluación
# ---------------------------------------------------------------------------

def obtener_metricas(nombre: str, y_true, y_pred) -> Dict[str, Any]:
    """Calcula métricas principales de rendimiento."""
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def obtener_confusion(nombre: str, y_true, y_pred) -> Dict[str, Any]:
    """Devuelve la matriz de confusión formateada."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


# ---------------------------------------------------------------------------
# Construcción del modelo
# ---------------------------------------------------------------------------

def crear_pipeline(vars_cat: List[str], vars_num: List[str]) -> GridSearchCV:
    """Define el pipeline completo y la búsqueda de hiperparámetros."""

    preprocesador = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), vars_cat),
            ("numeric", StandardScaler(), vars_num),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocessor", preprocesador),
            ("selector", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("mlp", MLPClassifier(max_iter=15000, random_state=21)),
        ]
    )

    grid = {
        "selector__k": [20],
        "pca__n_components": [None],
        "mlp__hidden_layer_sizes": [(50, 30, 40, 60)],
        "mlp__alpha": [0.26],
        "mlp__learning_rate_init": [0.001],
    }

    return GridSearchCV(
        estimator=modelo,
        param_grid=grid,
        cv=10,
        n_jobs=-1,
        scoring="balanced_accuracy",
        verbose=2,
        refit=True,
    )


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecución principal del script de modelado."""
    # Carga y limpieza
    train_df = preparar_datos(Paths.TRAIN)
    test_df = preparar_datos(Paths.TEST)

    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    num_features = [c for c in X_train.columns if c not in cat_features]

    # Construcción y entrenamiento
    grid_model = crear_pipeline(cat_features, num_features)
    grid_model.fit(X_train, y_train)

    # Guardar modelo
    limpiar_carpeta(Paths.MODELS)
    guardar_comprimido(grid_model, Paths.MODEL_FILE)

    # Predicciones
    y_train_pred = grid_model.predict(X_train)
    y_test_pred = grid_model.predict(X_test)

    # Cálculo y almacenamiento de métricas
    resultados = [
        obtener_metricas("train", y_train, y_train_pred),
        obtener_metricas("test", y_test, y_test_pred),
        obtener_confusion("train", y_train, y_train_pred),
        obtener_confusion("test", y_test, y_test_pred),
    ]

    registrar_metricas(resultados, Paths.METRICS_FILE)


if __name__ == "__main__":
    main()
