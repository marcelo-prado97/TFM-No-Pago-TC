import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
RUTA_DATOS = 'data/raw/UCI_Credit_Card.csv'
RUTA_MODELO = 'models/modelo_rf.pkl'
EXPERIMENTO = 'no_pago_tc'
UMBRAL_OPTIMO = 0.41


def dividir_datos(df: pd.DataFrame):
    X = df.drop(columns=['default.payment.next.month'])
    y = df['default.payment.next.month']
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
        "n_estimators": [64, 128, 512],
        "max_depth": [4, 6, 8, 10],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2", 0.3, 0.5]
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

    print('=' * 50)
    print('INICIO DEL PIPELINE DE ENTRENAMIENTO')
    print('=' * 50)

    # ==> PASO 1: Preprocesamiento
    print('\n[1/5] Iniciando preprocesamiento de datos...')
    df = preprocesar(RUTA_DATOS)
    x_train, x_test, y_train, y_test = dividir_datos(df)
    print(f'      Dataset cargado: {len(df)} registros')
    print(f'      Train: {len(x_train)} registros | Test: {len(x_test)} registros')
    print('      Preprocesamiento completado ✓')

    # ==> PASO 2: Configuracion MLflow
    print('\n[2/5] Configurando MLflow...')
    mlflow.set_tracking_uri("file:///" + os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'mlruns')
    ).replace("\\", "/"))
    mlflow.set_experiment(EXPERIMENTO)
    print('      MLflow configurado ✓')

    with mlflow.start_run():

        # ==> PASO 3: Entrenamiento del modelo
        print('\n[3/5] Entrenando modelo — buscando mejores hiperparámetros...')
        print('      Esto puede tardar varios minutos...')
        mejor_modelo, mejores_params = buscar_hiperparametros(x_train, y_train)
        print(f'      Mejores hiperparámetros encontrados: {mejores_params}')
        print('      Entrenamiento completado ✓')

        # ==> PASO 4: Evaluacion del modelo
        print('\n[4/5] Evaluando modelo sobre conjunto de test...')
        metricas = calcular_metricas(mejor_modelo, x_test, y_test)
        print(f'      AUC:       {metricas["auc"]:.4f}')
        print(f'      Accuracy:  {metricas["accuracy"]:.4f}')
        print(f'      Precision: {metricas["precision"]:.4f}')
        print(f'      Recall:    {metricas["recall"]:.4f}')
        print(f'      F1:        {metricas["f1"]:.4f}')
        print(f'      Umbral:    {UMBRAL_OPTIMO}')
        print('      Evaluación completada ✓')

        # Registrar en MLflow
        mlflow.log_params(mejores_params)
        mlflow.log_param('seed', SEED)
        mlflow.log_param('test_size', 0.3)
        mlflow.log_param('umbral', UMBRAL_OPTIMO)
        mlflow.log_param('dataset', 'UCI_Credit_Card.csv')
        mlflow.log_metric('auc', metricas['auc'])
        mlflow.log_metric('accuracy', metricas['accuracy'])
        mlflow.log_metric('precision', metricas['precision'])
        mlflow.log_metric('recall', metricas['recall'])
        mlflow.log_metric('f1', metricas['f1'])
        mlflow.sklearn.log_model(
            sk_model=mejor_modelo,
            artifact_path='modelo_rf'
        )

        # ==> PASO 5: Guardando modelo
        print('\n[5/5] Guardando modelo localmente...')
        os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
        joblib.dump(mejor_modelo, RUTA_MODELO)
        print(f'      Modelo guardado en: {RUTA_MODELO} ✓')
        print('      Experimento registrado en MLflow ✓')

    print('\n' + '=' * 50)
    print('PIPELINE FINALIZADO EXITOSAMENTE')
    print('=' * 50)


if __name__ == '__main__':
    entrenar()