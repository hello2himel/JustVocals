JustVocals
JustVocals is a Flask-based web application that extracts vocals from audio files or YouTube videos using AI-powered tools like Demucs. This guide explains how to build and run the application using Docker from this GitHub repository.
Prerequisites

Docker: Install Docker (e.g., sudo pacman -S docker on Arch Linux, sudo apt install docker.io on Ubuntu).
Git: Install Git (e.g., sudo pacman -S git or sudo apt install git).
Internet Access: Required to clone the repository and download dependencies.
Port Availability: Ensure port 5000 is free.
System Resources: At least 4GB RAM and a multi-core CPU recommended for Demucs.

Quick Start

Clone the Repository:
git clone https://github.com/yourusername/justvocals.git
cd justvocals


Create Directories:Create directories for persistent storage:
mkdir -p downloads separated final_output
chmod -R 777 downloads separated final_output


Build the Docker Image:
docker build -t justvocals .


Run the Container:
docker run -d -p 5000:5000 \
  -v $(pwd)/downloads:/app/downloads \
  -v $(pwd)/separated:/app/separated \
  -v $(pwd)/final_output:/app/final_output \
  -e SECRET_KEY=$(python3 -c "import os; print(os.urandom(24).hex())") \
  --name justvocals_container justvocals


Access the Application:Open a browser and navigate to http://localhost:5000.

Usage:

Upload an MP3/WAV file (max 100MB) or enter a YouTube URL.
Adjust settings (silence removal, vocal enhancement, silence threshold, etc.).
Download processed vocal tracks individually or as a ZIP.


Stop the Container:
docker stop justvocals_container
docker rm justvocals_container



Troubleshooting

Build Errors:
DNS Issues: If apt-get update fails, ensure internet access and try:docker build --build-arg DNS=8.8.8.8 -t justvocals .


Large Build Context: Ensure .dockerignore is present to exclude downloads, separated, and final_output.


Port Conflict: Use a different port (e.g., -p 8080:5000) and access http://localhost:8080.
Permission Issues: Verify directories have write permissions (chmod -R 777 downloads separated final_output).
Runtime Errors: Check logs:docker logs justvocals_container


Dependency Issues: Ensure ffmpeg and demucs are installed correctly in the container:docker run -it justvocals bash
ffmpeg -version
python -c "import demucs; print(demucs.__version__)"



Notes

The container uses a Debian-based image (python:3.10-slim) with ffmpeg and demucs.
Processed files are stored in the final_output directory on your host.
The application is set to Japan Standard Time (JST).
For production, the container uses gunicorn with eventlet.

Support
For issues, open a GitHub issue or contact the maintainer at [your contact info]. Support the project via https://ko-fi.com/hello2himel.