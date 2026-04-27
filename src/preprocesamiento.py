import pandas as pd
import numpy as np
import os

SEED = 876

COLUMNAS_ENTRADA = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'default.payment.next.month'
]


def cargar_datos(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    return df

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminar columna ID si existe
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Renombrar PAY_0 a PAY_1 para consistencia
    if 'PAY_0' in df.columns:
        df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Reclasificar categorias no documentadas en EDUCATION
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

    # Reclasificar categorias no documentadas en MARRIAGE
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    # Eliminar outlier extremo en BILL_AMT1 con criterio 3xIQR
    Q1 = df['BILL_AMT1'].quantile(0.25)
    Q3 = df['BILL_AMT1'].quantile(0.75)
    IQR = Q3 - Q1
    limite = Q3 + 3 * IQR
    df = df[df['BILL_AMT1'] <= limite].copy()

    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

    # Utilizacion del credito
    df['CREDIT_UTIL1'] = np.where(
        df['LIMIT_BAL'] == 0,
        0,
        df['BILL_AMT1'] / df['LIMIT_BAL']
    )

    # Variables de mora historica — clip(lower=0) ignora valores -2 y -1
    df['DEMORA_SUM'] = df[pay_cols].clip(lower=0).sum(axis=1)
    df['DEMORA_MAX'] = df[pay_cols].clip(lower=0).max(axis=1)

    # Meses al dia — PAY <= 0 significa sin mora activa
    df['MESES_AL_DIA'] = (df[pay_cols] <= 0).sum(axis=1)

    # Variacion de saldo entre el ultimo mes y el minimo de los 6 meses
    df['VARIACION_TOTAL_CUENTA'] = df['BILL_AMT1'] - df[bill_cols].min(axis=1)

    return df

def codificar_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nominal = ['MARRIAGE', 'SEX']
    df = pd.get_dummies(df, columns=nominal, drop_first=False, dtype=int)
    rename_dict = {
        "MARRIAGE_1": "Civil_Casado",
        "MARRIAGE_2": "Civil_Soltero",
        "MARRIAGE_3": "Civil_Otros",
        "SEX_1": "Sexo_Masculino",
        "SEX_2": "Sexo_Femenino"
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
    return df


def preprocesar(ruta: str) -> pd.DataFrame:
    df = cargar_datos(ruta)
    df = limpiar_datos(df)
    df = crear_features(df)
    df = codificar_variables(df) 
    return df