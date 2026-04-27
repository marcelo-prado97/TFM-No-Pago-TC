from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predecir

app = FastAPI(
    title='API - No Pago TC',
    description='Predicción de probabilidad de incumplimiento de pago para tarjetas de crédito — Dataset UCI Taiwan.',
    version='2.0.0'
)


class ClienteInput(BaseModel):
    LIMIT_BAL: float
    EDUCATION: int
    AGE: int
    PAY_1: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float
    CREDIT_UTIL1: float
    DEMORA_SUM: int
    DEMORA_MAX: int
    MESES_AL_DIA: int
    VARIACION_TOTAL_CUENTA: float
    Civil_Casado: int
    Civil_Soltero: int
    Civil_Otros: int
    Sexo_Masculino: int
    Sexo_Femenino: int


class PrediccionOutput(BaseModel):
    default_payment_next_month: int = Field(alias='default.payment.next.month')
    PROBABILIDAD_NOPAGO: float
    PROBABILIDAD_SIPAGO: float
    TASA_INTERES_MENSUAL: float
    TASA_INTERES_MORA: float
    model_config = {'populate_by_name': True}

@app.get('/')
def root():
    return {'mensaje': 'API No Pago TC activa. Visita /docs para ver la documentación.'}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predecir', response_model=List[PrediccionOutput])
def predecir_clientes(clientes: List[ClienteInput]):
    try:
        datos = [c.model_dump() for c in clientes]
        resultados = predecir(datos)
        return resultados
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predecir/individual', response_model=PrediccionOutput)
def predecir_cliente_individual(cliente: ClienteInput):
    try:
        datos = [cliente.model_dump()]
        resultado = predecir(datos)
        #return resultado[0]
        return PrediccionOutput(**resultado[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))