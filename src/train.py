import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn

from preprocesamiento import preprocesar

SEED = 876
RUTA_DATOS = 'data/raw/Data_TC.xlsx'
RUTA_MODELO = 'models/modelo_rf.pkl'
EXPERIMENTO = 'no_pago_tc'
UMBRAL_OPTIMO = 0.47


def dividir_datos(df: pd.DataFrame):
    X = df.drop(columns=['ES_NOPAGO'])
    y = df['ES_NOPAGO']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=SEED,
        stratify=y
    )
    return x_train, x_test, y_train, y_test


def buscar_hiperparametros(x_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(random_state=SEED, class_weight='balanced')

    params_rf = {
        'n_estimators': [16, 32, 64, 100],
        'max_depth': [7, 8, 9, 10, 11, 12, 13],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 6),
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_impurity_decrease': uniform(0.0, 0.01),
        'ccp_alpha': uniform(0.0, 0.002)
    }

    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=params_rf,
        n_iter=50,
        cv=5,
        scoring='roc_auc',
        random_state=SEED,
        n_jobs=-1
    )

    rf_random.fit(x_train, y_train)
    return rf_random.best_estimator_, rf_random.best_params_


def calcular_metricas(modelo, x_test, y_test, umbral=UMBRAL_OPTIMO) -> dict:
    y_proba = modelo.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)

    return {
        'auc': roc_auc_score(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'umbral': umbral
    }


def entrenar():
    # Preprocesamiento
    print('Cargando y preprocesando datos...')
    df = preprocesar(RUTA_DATOS)
    x_train, x_test, y_train, y_test = dividir_datos(df)

    # MLflow
    #mlflow.set_tracking_uri('mlruns')
    mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))

    mlflow.set_experiment(EXPERIMENTO)

    with mlflow.start_run():

        print('Buscando mejores hiperparámetros...')
        mejor_modelo, mejores_params = buscar_hiperparametros(x_train, y_train)

        print('Calculando métricas...')
        metricas = calcular_metricas(mejor_modelo, x_test, y_test)

        # Registrar parámetros
        mlflow.log_params(mejores_params)
        mlflow.log_param('seed', SEED)
        mlflow.log_param('test_size', 0.3)
        mlflow.log_param('umbral', UMBRAL_OPTIMO)

        # Registrar métricas
        mlflow.log_metric('auc', metricas['auc'])
        mlflow.log_metric('accuracy', metricas['accuracy'])
        mlflow.log_metric('precision', metricas['precision'])
        mlflow.log_metric('recall', metricas['recall'])
        mlflow.log_metric('f1', metricas['f1'])

        # Registrar modelo
        mlflow.sklearn.log_model(
            sk_model=mejor_modelo,
            artifact_path='modelo_rf',
            registered_model_name='rfclassmod_nopago'
        )

        # Guardar modelo localmente
        os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
        joblib.dump(mejor_modelo, RUTA_MODELO)

        print('Entrenamiento completado.')
        print(f"  AUC:       {metricas['auc']:.4f}")
        print(f"  Accuracy:  {metricas['accuracy']:.4f}")
        print(f"  Precision: {metricas['precision']:.4f}")
        print(f"  Recall:    {metricas['recall']:.4f}")
        print(f"  F1:        {metricas['f1']:.4f}")
        print(f"  Modelo guardado en: {RUTA_MODELO}")


if __name__ == '__main__':
    entrenar()