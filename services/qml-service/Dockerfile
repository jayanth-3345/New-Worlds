FROM python:3.12-slim
WORKDIR /app/services/qml-service/src
COPY services/qml-service/requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY services/qml-service/src/ ./
EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
