FROM python:3.10-slim

# Prevent interactive prompts & reduce layer size
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system deps needed for scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Copy only dependency list first for caching
COPY requirements.txt .

# Install python deps without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project
COPY . .

# CLEAN ANY HIDDEN GARBAGE (VERY IMPORTANT)
RUN rm -rf .git \
           **/.ipynb_checkpoints \
           **/__pycache__ \
           eda_outputs \
           yolo_dataset/images \
           yolo_dataset/labels

CMD ["python", "run_eda.py"]