FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN python -m pip install --no-cache-dir --upgrade pip

RUN python -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu

# Install PyTorch dependency
RUN python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    typing_extensions

# Install remaining project dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501