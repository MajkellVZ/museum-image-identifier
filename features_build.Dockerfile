FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY similarity_finder.py .
COPY gcp_build_features.py .

CMD ["python", "gcp_build_features.py"]