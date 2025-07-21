# Optimized Dockerfile for Google Cloud Run GPU - Fashion ML API
# Deploy directly from GitHub to Cloud Run
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directory
RUN mkdir -p /app/.cache

# Pre-download models during build time to reduce cold start
RUN python -c "
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel
import torch

print('🔽 Downloading models for faster cold starts...')

# Set cache directory
os.environ['TRANSFORMERS_CACHE'] = '/app/.cache'
os.environ['HF_HOME'] = '/app/.cache'

try:
    # Download Image Captioning Model
    print('📷 Downloading image captioning model...')
    try:
        # Try the custom FashionBLIP model first
        processor = BlipProcessor.from_pretrained('rcfg/FashionBLIP-1', cache_dir='/app/.cache')
        model = BlipForConditionalGeneration.from_pretrained('rcfg/FashionBLIP-1', cache_dir='/app/.cache')
        print('✅ FashionBLIP model downloaded')
    except:
        # Fallback to base BLIP model
        print('📷 Using base BLIP model as fallback...')
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir='/app/.cache')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir='/app/.cache')
        print('✅ Base BLIP model downloaded')
    
    # Download Text Vectorization Model
    print('📝 Downloading text vectorization model...')
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/.cache')
    text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/.cache')
    print('✅ Text vectorization model downloaded')
    
    print('🎉 All models downloaded successfully!')
    
except Exception as e:
    print(f'⚠️ Error downloading models: {e}')
    print('Models will be downloaded at runtime')
"

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-5000} 