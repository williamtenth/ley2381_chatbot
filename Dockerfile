# Usar imagen Python slim (mucho más pequeña)
FROM python:3.11-slim

# Configurar directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (mínimas)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Variables de entorno
ENV PYTHONPATH=/app
ENV PORT=8000

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar
CMD ["python", "-m", "src.simple_mcp_server"]
