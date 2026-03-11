import joblib
import pandas as pd
import numpy as np

RUTA_MODELO = 'models/modelo_rf.pkl'
UMBRAL_OPTIMO = 0.47

COLUMNAS_ESPERADAS = [
    'MONT_CREDIT', 'NIV_EDUC', 'AGE',
    'PAY_1', 'PAY_2', 'PAY_3',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'CREDIT_UTIL1', 'DEMORA_SUM', 'DEMORA_MAX',
    'MESES_AL_DIA', 'VARIACION_TOTAL_CUENTA',
    'Civil_Casado', 'Civil_Soltero', 'Civil_Otros',
    'Sexo_Masculino', 'Sexo_Femenino'
]


def cargar_modelo(ruta: str = RUTA_MODELO):
    return joblib.load(ruta)


def predecir(datos: list[dict], umbral: float = UMBRAL_OPTIMO) -> list[dict]:
    modelo = cargar_modelo()

    df = pd.DataFrame(datos)

    # Alinear columnas con las que espera el modelo
    for col in COLUMNAS_ESPERADAS:
        if col not in df.columns:
            df[col] = 0
    df = df[COLUMNAS_ESPERADAS]

    # Predicciones
    probabilidades = modelo.predict_proba(df)
    prob_nopago = probabilidades[:, 1]
    prob_sipago = probabilidades[:, 0]
    es_nopago = (prob_nopago >= umbral).astype(int)

    # Calcular tasa de interes mora segun nivel de riesgo
    tasas = [calcular_tasa_mora(p) for p in prob_nopago]

    resultados = []
    for i in range(len(df)):
        resultados.append({
            'ES_NOPAGO': int(es_nopago[i]),
            'PROBABILIDAD_NOPAGO': round(float(prob_nopago[i]), 6),
            'PROBABILIDAD_SIPAGO': round(float(prob_sipago[i]), 6),
            'TASA_INTERES_MENSUAL': 0.10,
            'TASA_INTERES_MORA': tasas[i]
        })

    return resultados


def calcular_tasa_mora(probabilidad_nopago: float) -> float:
    if probabilidad_nopago < 0.50:
        return 0.0
    elif probabilidad_nopago < 0.60:
        return 0.02
    elif probabilidad_nopago < 0.70:
        return 0.04
    elif probabilidad_nopago < 0.80:
        return 0.06
    elif probabilidad_nopago < 0.90:
        return 0.08
    else:
        return 0.10


if __name__ == '__main__':
    # Prueba rapida con los datos de ejemplo del notebook
    datos_prueba = [
        {
            'MONT_CREDIT': 5000, 'NIV_EDUC': 2, 'AGE': 25,
            'PAY_1': 0, 'PAY_2': 1, 'PAY_3': 0,
            'BILL_AMT1': 4500, 'BILL_AMT2': 4000, 'BILL_AMT3': 3500,
            'PAY_AMT1': 2000, 'PAY_AMT2': 1000, 'PAY_AMT3': 1200,
            'CREDIT_UTIL1': 0.75, 'DEMORA_SUM': 12, 'DEMORA_MAX': 2,
            'MESES_AL_DIA': 2, 'VARIACION_TOTAL_CUENTA': 1200,
            'Civil_Casado': 1, 'Civil_Soltero': 0, 'Civil_Otros': 0,
            'Sexo_Masculino': 1, 'Sexo_Femenino': 0
        },
        {
            'MONT_CREDIT': 2000, 'NIV_EDUC': 1, 'AGE': 45,
            'PAY_1': 2, 'PAY_2': 2, 'PAY_3': 1,
            'BILL_AMT1': 3000, 'BILL_AMT2': 2800, 'BILL_AMT3': 2600,
            'PAY_AMT1': 800, 'PAY_AMT2': 700, 'PAY_AMT3': 600,
            'CREDIT_UTIL1': 0.90, 'DEMORA_SUM': 5, 'DEMORA_MAX': 2,
            'MESES_AL_DIA': 0, 'VARIACION_TOTAL_CUENTA': 400,
            'Civil_Casado': 0, 'Civil_Soltero': 1, 'Civil_Otros': 0,
            'Sexo_Masculino': 0, 'Sexo_Femenino': 1
        }
    ]

    resultados = predecir(datos_prueba)
    for i, r in enumerate(resultados):
        print(f'Cliente {i + 1}:')
        print(f"  ES_NOPAGO:           {r['ES_NOPAGO']}")
        print(f"  PROBABILIDAD_NOPAGO: {r['PROBABILIDAD_NOPAGO']}")
        print(f"  PROBABILIDAD_SIPAGO: {r['PROBABILIDAD_SIPAGO']}")
        print(f"  TASA_INTERES_MORA:   {r['TASA_INTERES_MORA']}")
        print()