FROM python:3.9-slim

# Install system dependencies for GDAL/GeoPandas
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy dependencies
COPY requirements.txt .

# Install dependencies (use --no-cache-dir to keep image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port (default 5000)
EXPOSE 5000

# Default command (uses Gunicorn for production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
