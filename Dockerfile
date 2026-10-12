FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY feast_repo ./feast_repo
COPY data/phase3/train_combined_features.csv ./data/phase3/train_combined_features.csv
COPY data/phase3/validation_combined_features.csv ./data/phase3/validation_combined_features.csv
RUN rm -f /app/feast_repo/fraud_features/feature_repo/data/registry.db /app/feast_repo/fraud_features/feature_repo/data/online_store.db && cd /app/feast_repo/fraud_features/feature_repo && feast apply

RUN mkdir -p /app/artifacts/rf300_champion

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
