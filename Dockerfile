# Multi-stage Dockerfile for YMERA Enterprise Platform
# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source
COPY *.ts* ./
COPY src ./src
COPY shared ./shared

# Build frontend
RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python application
COPY *.py ./
COPY app ./app

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/dist ./static

# Create necessary directories
RUN mkdir -p /app/storage /app/logs /app/uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["python", "main_production.py"]
