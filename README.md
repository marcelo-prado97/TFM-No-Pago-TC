# 🏦 Predicción de No Pago en Tarjetas de Crédito
### TFM — Marco Metodológico Híbrido CRISP-ML(Q) + Scrum + MLOps

**Autor:** Enrique Calle Prado
**Universidad:** Universidad Internacional de Valencia (VIU)
**Máster:** Big Data y Data Science
**Director:** Dr. Horacio Daniel Kuna

---

## 📋 Descripción

Validación práctica del marco metodológico híbrido **CRISP-ML(Q) + Scrum + MLOps** mediante un caso de estudio de predicción de incumplimiento de pago en tarjetas de crédito (dataset Taiwan, Yeh & Lien, 2009).

El modelo predice la **probabilidad de no pago** de un cliente para permitir el ajuste dinámico de tasas de interés según nivel de riesgo crediticio.

---

## 🏗️ Estructura del Proyecto

```
Desarrollo TC Default/
├── .github/workflows/     # CI/CD con GitHub Actions
├── api/                   # Endpoint FastAPI para predicciones
├── data/
│   ├── raw/               # Dataset original (versionado con DVC)
│   └── processed/         # Datos procesados
├── models/                # Modelos serializados (versionados con DVC)
├── monitoring/            # Reportes de drift con Evidently AI
├── notebooks/             # Exploración y desarrollo
│   └── no_pago_tc.ipynb
├── src/                   # Código fuente modularizado
│   ├── train.py           # Pipeline de entrenamiento
│   └── predict.py         # Lógica de predicción
├── tests/                 # Pruebas automatizadas
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 🔄 Sprints del Proyecto (CRISP-ML(Q) + Scrum)

### Sprint 1 — Comprensión del negocio y de los datos
> *Fases CRISP-ML(Q): Fase 1 + Fase 2*

- Definición de objetivos del proyecto y criterios de éxito
- Análisis Exploratorio de Datos (EDA)
- Evaluación de calidad de datos y distribución de clases
- **Herramientas:** Git, Jupyter, pandas, seaborn

---

### Sprint 2 — Preparación de datos y Modelado (inicio)
> *Fases CRISP-ML(Q): Fase 3 + Fase 4 (inicio)*

- Limpieza de datos y feature engineering
- Tratamiento del desbalanceo de clases
- Versionado del dataset con DVC
- Selección de algoritmos y entrenamiento inicial (RF, KNN, LogReg)
- **Herramientas:** DVC, scikit-learn, pandas, numpy

---

### Sprint 3 — Modelado (cierre) y Evaluación
> *Fases CRISP-ML(Q): Fase 4 (cierre) + Fase 5*

- Tracking de experimentos con MLflow
- Optimización de hiperparámetros (RandomizedSearchCV)
- Selección del modelo final (Random Forest, AUC=0.9355)
- Validación cruzada y análisis de errores
- **Herramientas:** MLflow, scikit-learn, joblib

---

### Sprint 4 — Despliegue y Monitoreo (inicio)
> *Fases CRISP-ML(Q): Fase 6 + Fase 7 (inicio)*

- Desarrollo de API REST con FastAPI
- Containerización con Docker
- Configuración de CI/CD con GitHub Actions
- Implementación inicial de Evidently AI
- **Herramientas:** FastAPI, Docker, GitHub Actions, Evidently AI

---

### Sprint 5 — Monitoreo (cierre) y Cierre del proyecto
> *Fases CRISP-ML(Q): Fase 7 (cierre)*

- Detección de data drift y generación de alertas
- Evaluación final del pipeline MLOps (latencia, throughput, reproducibilidad)
- Documentación del marco híbrido
- **Herramientas:** Evidently AI, pytest

---

## 🤖 Modelos Evaluados

| Modelo | AUC-ROC | F1-Score | Umbral |
|---|---|---|---|
| **Random Forest** ✅ | 0.9355 | 0.8353 | 0.47 |
| KNN | 0.9174 | 0.8019 | 0.44 |
| Logistic Regression | 0.8194 | 0.6892 | 0.41 |

**Modelo seleccionado:** Random Forest Classifier

---

## 🚀 Cómo ejecutar

### 1. Clonar el repositorio
```bash
git clone <url-del-repo>
cd "Desarrollo TC Default"
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv entorno
entorno\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Restaurar datos con DVC
```bash
dvc pull
```

### 4. Ejecutar la API localmente
```bash
uvicorn api.main:app --reload
```
La API estará disponible en: `http://localhost:8000/docs`

### 5. Ejecutar con Docker
```bash
docker build -t no-pago-tc-api .
docker run -p 8000:8000 no-pago-tc-api
```

---

## 📊 Stack Tecnológico (MLOps)

| Herramienta | Sprint | Rol |
|---|---|---|
| **Git** | 1-5 | Control de versiones del código |
| **DVC** | 2 | Versionado de datos y modelos |
| **MLflow** | 3 | Tracking de experimentos y métricas |
| **FastAPI** | 4 | Despliegue del modelo como API REST |
| **Docker** | 4 | Contenedorización y reproducibilidad |
| **GitHub Actions** | 4 | CI/CD automatizado |
| **Evidently AI** | 4-5 | Monitoreo de data drift en producción |
| **GitHub Projects** | 1-5 | Gestión ágil con Scrum (Kanban) |

---

## 📡 Endpoint de Predicción

**POST** `/predict`

```json
{
  "MONT_CREDIT": 5000,
  "NIV_EDUC": 2,
  "AGE": 25,
  "PAY_1": 0,
  "PAY_2": 1,
  "PAY_3": 0,
  "BILL_AMT1": 4500,
  "BILL_AMT2": 4000,
  "BILL_AMT3": 3500,
  "PAY_AMT1": 2000,
  "PAY_AMT2": 1000,
  "PAY_AMT3": 1200
}
```

**Respuesta:**
```json
{
  "ES_NOPAGO": 1,
  "PROBABILIDAD_SIPAGO": 0.307,
  "PROBABILIDAD_NOPAGO": 0.693,
  "TASA_INTERES_MORA": 0.04
}
```

---

## 📅 Fechas Clave

| Hito | Fecha |
|---|---|
| Entrega borrador de memoria | 12 de marzo de 2026 |
| Depósito TFM | 2 de abril de 2026 |
| Defensa TFM | 20-30 de abril de 2026 |
