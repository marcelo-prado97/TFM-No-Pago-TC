# Imagen base
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer el puerto de la API
EXPOSE 8000

# Comando para levantar la API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]