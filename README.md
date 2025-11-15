# Bosch Assignment - EDA container

## Run EDA in Docker
1. Build image:
   `docker build -t bosch-eda:latest .`
2. Run (mount your local dataset):
   Windows (PowerShell):
   `docker run --rm -v "C:\Users\niran\OneDrive\Desktop\Bosch\Assignment\EDA\data:/app/data:ro" -v "%CD%/eda_outputs:/app/eda_outputs" bosch-eda:latest`
   OR
   `docker-compose up --build`
3. Outputs will be in `./eda_outputs/` on host.

## Notes
- `data/` is intentionally not tracked by git. Mount the dataset into `/app/data`.
- For training containers (GPU), build a separate image on a machine with NVIDIA Docker and CUDA drivers.
