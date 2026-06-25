FROM python:3.11-slim

WORKDIR /app

# Pure-Python deps (trafilatura, pypdf, httpx, mcp) need no system packages.

# Dependencies as a separate layer: change less often than code → cached better
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code (stateless service — no data/ or templates/)
COPY src/ src/
COPY main.py .

# No EXPOSE: the service is published by Traefik via docker-compose labels.

CMD ["python", "main.py"]
