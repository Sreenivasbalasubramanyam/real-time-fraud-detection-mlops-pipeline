# Dockerfile for the real-time fraud scoring consumer service.
#
# This image runs `src/streaming/consumer.py` in "production mode"
# (KAFKA_MODE=real), connecting to the Kafka broker defined in
# docker-compose.yml and scoring each incoming payment transaction with the
# trained XGBoost model. See README.md "Production / Docker Compose Mode".

FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost / scientific python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir kafka-python==2.0.2

COPY . .

ENV KAFKA_MODE=real
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV PYTHONUNBUFFERED=1

# Entrypoint: run the scoring consumer against the real Kafka backend.
# (Requires a trained model under models/ — mount it or bake it into the
# image at build time via `python src/models/train_model.py` in CI.)
CMD ["python", "-m", "src.streaming.consumer"]
