import joblib, sys, os
sys.path.append('src')
from preprocesamiento import preprocesar
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

modelo = joblib.load('models/modelo_rf.pkl')
df = preprocesar('data/raw/Data_TC.xlsx')

TARGET = 'ES_NOPAGO'
UMBRAL = 0.47
features = [col for col in df.columns if col != TARGET]

ref, prod = train_test_split(df, test_size=0.3, random_state=876, stratify=df[TARGET])

y_pred = (modelo.predict_proba(prod[features])[:, 1] >= UMBRAL).astype(int)
print("=== METRICAS REALES EN PRODUCCION ===")
print(classification_report(prod[TARGET], y_pred))
print(f"Distribucion predicciones: {y_pred.sum()} unos de {len(y_pred)} total")
print(f"Distribucion real: {prod[TARGET].sum()} unos de {len(prod[TARGET])} total")