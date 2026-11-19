FROM python:3.12-slim
WORKDIR /app/services/llm-benchmark-service/src
COPY services/llm-benchmark-service/src/ ./
RUN pip install --no-cache-dir fastapi uvicorn pydantic groq python-dotenv
EXPOSE 8005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
