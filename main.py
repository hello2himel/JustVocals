import os
import shutil
import time
import logging

from flask import Flask
from flask_socketio import SocketIO

from config import Config, LOCAL_IP
from routes import register_routes


logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins=Config.CORS_ORIGINS, logger=True, engineio_logger=True)

# Register all routes and SocketIO handlers
register_routes(app, socketio)


def cleanup_old_sessions():
    base_dir = Config.BASE_DIR
    if not os.path.exists(base_dir):
        return

    now = time.time()
    timeout_seconds = Config.SESSION_TIMEOUT_HOURS * 3600

    for session_dir in os.listdir(base_dir):
        session_path = os.path.join(base_dir, session_dir)
        if os.path.isdir(session_path):
            try:
                mtime = os.path.getmtime(session_path)
                if now - mtime > timeout_seconds:
                    shutil.rmtree(session_path)
                    logger.info(f"Cleaned up old session: {session_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session_dir}: {str(e)}")


if __name__ == '__main__':
    cleanup_old_sessions()
    logger.info(f"Server starting...\n"
                f"🌐 Localhost: http://localhost:5000\n"
                f"📡 Local Network: http://{LOCAL_IP}:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)