FROM python:3.11-slim

WORKDIR /app

# Pure-Python deps (trafilatura, pypdf, httpx, mcp) need no system packages.

# Dependencies as a separate layer: change less often than code → cached better
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime state directory (persistent log file lives here; mounted as a volume)
RUN mkdir -p data

# Code
COPY src/ src/
COPY main.py .

# No EXPOSE: the service is published by Traefik via docker-compose labels.

CMD ["python", "main.py"]
