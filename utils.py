import os
import tempfile
import time
import subprocess
import threading
import logging
from contextlib import contextmanager
from functools import lru_cache

import librosa

from config import Config


logger = logging.getLogger(__name__)


# Custom exceptions
class AudioProcessingError(Exception):
    pass


class DownloadError(Exception):
    pass


class ValidationError(Exception):
    pass


def get_user_directories(session_id):
    base_dir = os.path.join(Config.BASE_DIR, session_id)
    return {
        'download': os.path.join(base_dir, 'downloads'),
        'separated': os.path.join(base_dir, 'separated'),
        'final': os.path.join(base_dir, 'final_output')
    }


@contextmanager
def temp_audio_file(suffix='.wav'):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp_file.name
    finally:
        try:
            os.remove(temp_file.name)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_file.name}: {str(e)}")


def sanitize_filename(filename):
    filename = os.path.basename(filename)
    filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
    if len(filename) > 255:
        filename = filename[:255]
    if not any(filename.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
        raise ValidationError(f"Invalid file extension for {filename}")
    return filename


def validate_audio_file_fast(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if not header.startswith((b'RIFF', b'ID3', b'\xff\xfb')):
                raise ValidationError("Invalid audio file format")
        return True
    except Exception as e:
        raise ValidationError(f"File validation failed: {str(e)}")


@lru_cache(maxsize=32)
def get_audio_metadata(file_path):
    try:
        duration = librosa.get_duration(path=file_path)
        return {'duration': duration}
    except Exception as e:
        logger.warning(f"Failed to get metadata for {file_path}: {str(e)}")
        return {'duration': 0}


def emit_log(socketio, message, type="info", error_context=None, sid=None):
    socketio.emit('log_message', {'message': message, 'type': type}, room=sid)
    log_level = 'info' if type == 'success' else type
    if type == "error" and error_context:
        logger.error(f"{message} | Context: {error_context}", exc_info=True)
    else:
        getattr(logger, log_level)(message)


def emit_progress(socketio, file_index, total_files, step, total_steps, stage_name="Processing", sid=None):
    progress_per_file = 100 / total_files
    progress_per_step_in_file = progress_per_file / total_steps
    current_file_base_progress = (file_index - 1) * progress_per_file
    current_step_progress = step * progress_per_step_in_file
    total_progress = current_file_base_progress + current_step_progress
    socketio.emit('progress_update', {'progress': round(total_progress, 1), 'stage': stage_name}, room=sid)


def progress_heartbeat(socketio, process_name, stop_event, sid=None):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        emit_log(socketio, f"⏳ {process_name} in progress ({elapsed:.1f}s elapsed)...", "info", sid=sid)
        time.sleep(Config.LOG_UPDATE_INTERVAL)


def run_subprocess_with_timeout(command, timeout=Config.MAX_PROCESSING_TIME, progress_callback=None, sid=None):
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        start_time = time.time()
        stderr_output = []

        def monitor_progress():
            while process.poll() is None:
                line = process.stderr.readline()
                if line:
                    stderr_output.append(line)
                    if progress_callback and '|' in line:
                        try:
                            progress = line.split('|')[1].split('/')[0].strip()
                            progress = float(progress) / 35.1 * 100
                            progress_callback(round(progress, 1))
                        except (IndexError, ValueError):
                            pass
                if time.time() - start_time > timeout:
                    process.terminate()
                    raise AudioProcessingError(f"Subprocess timed out after {timeout} seconds")
                time.sleep(0.1)

        if progress_callback:
            monitor_thread = threading.Thread(target=monitor_progress)
            monitor_thread.daemon = True
            monitor_thread.start()

        stdout, stderr = process.communicate(timeout=timeout)
        stderr_output.append(stderr)
        if process.returncode != 0:
            raise AudioProcessingError(f"Subprocess failed: {''.join(stderr_output)}")
        return subprocess.CompletedProcess(command, process.returncode, stdout, ''.join(stderr_output))
    except subprocess.TimeoutExpired:
        process.terminate()
        raise AudioProcessingError(f"Subprocess timed out after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(f"Subprocess failed: {e.stderr}")
    except Exception as e:
        raise AudioProcessingError(f"Subprocess error: {str(e)}")
