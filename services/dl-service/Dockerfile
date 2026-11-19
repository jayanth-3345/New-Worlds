FROM python:3.12-slim
WORKDIR /app/services/dl-service
COPY services/dl-service/requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "torch==2.7.0+cpu" --extra-index-url https://pypi.org/simple --index-url https://download.pytorch.org/whl/cpu
COPY services/dl-service/src/ ./src/
COPY services/dl-service/models/ ./models/
WORKDIR /app/services/dl-service/src
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
