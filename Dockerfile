# Use Debian-based Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Configure Debian repositories and install system dependencies
RUN echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to JST
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Demucs
RUN pip install --no-cache-dir demucs

# Expose port
EXPOSE 5000

# Create directories
RUN mkdir -p /app/downloads /app/separated /app/final_output

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Command to run (using gunicorn with gevent for production)
CMD ["gunicorn", "--worker-class", "gevent", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]