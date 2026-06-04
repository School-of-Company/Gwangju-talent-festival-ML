FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY trainer/ ./trainer/

ENV MODEL_PATH=./models/iforest-v1/model.joblib

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
