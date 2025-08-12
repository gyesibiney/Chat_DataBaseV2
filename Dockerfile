# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps if needed (sqlite is included)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# expose the uvicorn port
EXPOSE 8000

# Use uvicorn to run the FastAPI app
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
