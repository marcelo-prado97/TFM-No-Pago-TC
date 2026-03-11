import pandas as pd
import numpy as np


SEED = 876

COLUMNAS_ENTRADA = [
    'MONT_CREDIT', 'SEX', 'NIV_EDUC', 'EST_CIVIL', 'AGE',
    'PAY_1', 'PAY_2', 'PAY_3',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'ES_NOPAGO'
]


def cargar_datos(ruta: str) -> pd.DataFrame:
    df = pd.read_excel(ruta)
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminar columna ID si existe
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Eliminar outlier extremo en BILL_AMT1
    df = df[df['BILL_AMT1'] > -159521.0].copy()

    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Utilizacion del credito
    df['CREDIT_UTIL1'] = np.where(
        df['MONT_CREDIT'] == 0,
        0,
        df['BILL_AMT1'] / df['MONT_CREDIT']
    )

    # Variables de mora historica
    df['DEMORA_SUM'] = df[['PAY_1', 'PAY_2', 'PAY_3']].sum(axis=1)
    df['DEMORA_MAX'] = df[['PAY_1', 'PAY_2', 'PAY_3']].max(axis=1)
    df['MESES_AL_DIA'] = (df[['PAY_1', 'PAY_2', 'PAY_3']] == 0).sum(axis=1)

    # Variacion de saldo entre el ultimo mes y el minimo de los 3 meses
    df['VARIACION_TOTAL_CUENTA'] = df['BILL_AMT1'] - df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']].min(axis=1)

    return df


def codificar_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-Hot Encoding para variables nominales
    nominal = ['EST_CIVIL', 'SEX']
    df = pd.get_dummies(df, columns=nominal, drop_first=False, dtype=int)

    # Renombrar columnas para claridad
    rename_dict = {
        'EST_CIVIL_1': 'Civil_Casado',
        'EST_CIVIL_2': 'Civil_Soltero',
        'EST_CIVIL_3': 'Civil_Otros',
        'SEX_1': 'Sexo_Masculino',
        'SEX_2': 'Sexo_Femenino'
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    return df


def preprocesar(ruta: str) -> pd.DataFrame:
    df = cargar_datos(ruta)
    df = limpiar_datos(df)
    df = crear_features(df)
    df = codificar_variables(df)
    return df