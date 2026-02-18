import os
import secrets
import socket
import logging


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(16))
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_MB', 100)) * 1024 * 1024
    BASE_DIR = os.environ.get('BASE_DIR', 'user_data')
    SESSION_TIMEOUT_HOURS = int(os.environ.get('SESSION_TIMEOUT_HOURS', 24))
    ALLOWED_EXTENSIONS = {'.mp3', '.wav'}
    MAX_PROCESSING_TIME = int(os.environ.get('MAX_PROCESSING_TIME', 1800))  # 30 minutes
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    THREAD_WORKERS = min(int(os.environ.get('THREAD_WORKERS', 4)), os.cpu_count() or 4)
    SILENCE_THRESH_DEFAULT = int(os.environ.get('SILENCE_THRESH_DEFAULT', -40))
    MIN_SILENCE_LEN_DEFAULT = int(os.environ.get('MIN_SILENCE_LEN_DEFAULT', 2000))
    KEEP_SILENCE_DEFAULT = int(os.environ.get('KEEP_SILENCE_DEFAULT', 500))
    CORS_ORIGINS = [
        'http://localhost:5000',
        'http://127.0.0.1:5000',
        'http://192.168.0.101:5000',
    ]
    LOG_UPDATE_INTERVAL = float(os.environ.get('LOG_UPDATE_INTERVAL', 2.0))
    COOKIES_FILE = os.environ.get('COOKIES_FILE', '')
    COOKIES_FROM_BROWSER = os.environ.get('COOKIES_FROM_BROWSER', '')


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vocal_extractor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Determine local IP for CORS
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    LOCAL_IP = s.getsockname()[0]
    s.close()
    Config.CORS_ORIGINS.append(f'http://{LOCAL_IP}:5000')
except Exception as e:
    logger.warning(f"Could not determine local IP: {str(e)}")
    LOCAL_IP = "unknown"

Config.CORS_ORIGINS = list(set(Config.CORS_ORIGINS))
logger.info(f"CORS allowed origins: {Config.CORS_ORIGINS}")
