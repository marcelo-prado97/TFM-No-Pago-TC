import pandas as pd
import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocesamiento import preprocesar

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping
from sklearn.model_selection import train_test_split

# ==> Cargar modelo
modelo = joblib.load('models/modelo_rf.pkl')

# ==> Cargar y preprocesar dataset
df = preprocesar('data/raw/UCI_Credit_Card.csv')

# ==> Definir columnas
TARGET = 'default.payment.next.month'
UMBRAL = 0.41
features = [col for col in df.columns if col != TARGET]


X = df[features]
y = df[TARGET]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=876,
    stratify=y
)

referencia = X_train.copy()
referencia[TARGET] = y_train.values

produccion = X_test.copy()
produccion[TARGET] = y_test.values


# # ==> Dividir aleatoriamente con estratificacion
# referencia, produccion = train_test_split(
#     df,
#     test_size=0.3,
#     random_state=876,
#     stratify=df[TARGET]
# )
# referencia = referencia.copy()
# produccion = produccion.copy()




# ==> Agregar probabilidades y predicciones con umbral correcto
referencia['prediction'] = (
    modelo.predict_proba(referencia[features])[:, 1] >= UMBRAL
).astype(int)

produccion['prediction'] = (
    modelo.predict_proba(produccion[features])[:, 1] >= UMBRAL
).astype(int)

# ==> Column mapping
column_mapping = ColumnMapping(
    target=TARGET,
    prediction='prediction',
    pos_label=1,
    numerical_features=[col for col in features 
                        if col not in ['Civil_Casado', 'Civil_Soltero',
                                      'Civil_Otros', 'Sexo_Masculino', 'Sexo_Femenino',
                                      'EDUCATION']],
    categorical_features=['Civil_Casado', 'Civil_Soltero', 'Civil_Otros',
                          'Sexo_Masculino', 'Sexo_Femenino', 'EDUCATION']
)
from sklearn.metrics import recall_score, f1_score, roc_auc_score
y_true_prod = produccion[TARGET]
y_pred_prod = produccion['prediction']
y_proba_prod = modelo.predict_proba(produccion[features])[:, 1]
print('=== METRICAS EN PRODUCCION ===')
print(f'AUC:       {roc_auc_score(y_true_prod, y_proba_prod):.4f}')
print(f'Recall:    {recall_score(y_true_prod, y_pred_prod):.4f}')
print(f'F1:        {f1_score(y_true_prod, y_pred_prod):.4f}')


# ==> Generar reporte
reporte = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

reporte.run(
    reference_data=referencia,
    current_data=produccion,
    column_mapping=column_mapping
)

# ==> Guardar reporte HTML
reporte.save_html('monitoring/reporte_monitoreo.html')
print("Reporte generado: monitoring/reporte_monitoreo.html")