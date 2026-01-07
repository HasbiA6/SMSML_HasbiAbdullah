FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py /app/inference.py
COPY prometheus_exporter.py /app/prometheus_exporter.py

EXPOSE 8000 9000