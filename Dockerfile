FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY models ./models

RUN mkdir -p lime_results

CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}