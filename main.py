import os
import shutil
import subprocess
import threading
import json
import logging
import secrets
import uuid
import time
from contextlib import contextmanager
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import tempfile
import socket
from flask import Flask, render_template, request, send_from_directory, Response, session, abort, send_file
from flask_socketio import SocketIO, emit, join_room
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, FileField
from wtforms.validators import Regexp, Optional
import yt_dlp
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt


# Configuration
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

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins=Config.CORS_ORIGINS, logger=True, engineio_logger=True)

# Thread pool
executor = ThreadPoolExecutor(max_workers=Config.THREAD_WORKERS)

# Active downloads tracking
active_downloads = set()
download_lock = threading.Lock()


# Custom exceptions
class AudioProcessingError(Exception): pass


class DownloadError(Exception): pass


class ValidationError(Exception): pass


# Helper functions
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


def emit_log(message, type="info", error_context=None, sid=None):
    socketio.emit('log_message', {'message': message, 'type': type}, room=sid)
    log_level = 'info' if type == 'success' else type
    if type == "error" and error_context:
        logger.error(f"{message} | Context: {error_context}", exc_info=True)
    else:
        getattr(logger, log_level)(message)


def emit_progress(file_index, total_files, step, total_steps, stage_name="Processing", sid=None):
    progress_per_file = 100 / total_files
    progress_per_step_in_file = progress_per_file / total_steps
    current_file_base_progress = (file_index - 1) * progress_per_file
    current_step_progress = step * progress_per_step_in_file
    total_progress = current_file_base_progress + current_step_progress
    socketio.emit('progress_update', {'progress': round(total_progress, 1), 'stage': stage_name}, room=sid)


def progress_heartbeat(process_name, stop_event, sid=None):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        emit_log(f"⏳ {process_name} in progress ({elapsed:.1f}s elapsed)...", "info", sid=sid)
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


# Voice Enhancement Logic
def enhance_vocals(vocals, sr):
    try:
        max_val = np.max(np.abs(vocals))
        if max_val > 0:
            vocals = vocals / max_val * 0.95

        sos = butter(4, 80, btype='high', fs=sr, output='sos')
        vocals = sosfilt(sos, vocals)

        def soft_compress(audio, threshold=0.3, ratio=3.0):
            compressed = np.copy(audio)
            mask = np.abs(audio) > threshold
            excess = np.abs(audio[mask]) - threshold
            compressed[mask] = np.sign(audio[mask]) * (threshold + excess / ratio)
            return compressed

        vocals = soft_compress(vocals)
        return vocals
    except Exception as e:
        emit_log(f"⚠️ Enhancement failed, using original: {str(e)}", "warning")
        return vocals


# Silence Removal Logic
def remove_silence(audio_path, output_path, silence_thresh, min_silence_len, keep_silence, sid=None):
    filename = os.path.basename(audio_path)
    emit_log(f"🔇 Processing silence removal for {filename}...", "info", sid=sid)

    try:
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        if file_size > 100:
            emit_log(f"⚠️ Large file ({file_size:.1f}MB), processing may take time", "warning", sid=sid)

        audio, sr = librosa.load(audio_path, sr=None)
        total_samples = len(audio)
        emit_log(f"📈 Loaded audio: {total_samples / sr:.1f}s, {sr}Hz", "info", sid=sid)

        if sr < 8000 or sr > 192000:
            emit_log(f"⚠️ Unusual sample rate: {sr}Hz", "warning", sid=sid)

        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(frame_length // 2)
        min_silence_samples = int(min_silence_len * sr / 1000)
        keep_silence_samples = int(keep_silence * sr / 1000)

        emit_log("🔄 Computing audio energy...", "info", sid=sid)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        max_rms = np.max(rms)

        if max_rms == 0:
            emit_log("⚠️ Audio is completely silent, keeping original", "warning", sid=sid)
            shutil.copy(audio_path, output_path)
            return True

        silence_thresh_linear = 10 ** (silence_thresh / 20)
        dynamic_thresh = max(silence_thresh_linear, max_rms * 0.05)
        emit_log(f"🔍 Using silence threshold: {20 * np.log10(dynamic_thresh):.1f}dB", "info", sid=sid)

        silent_frames = rms < dynamic_thresh
        if len(silent_frames) == 0:
            emit_log("⚠️ No frames detected, keeping original", "warning", sid=sid)
            shutil.copy(audio_path, output_path)
            return True

        frame_times = librosa.frames_to_samples(np.arange(len(silent_frames)), hop_length=hop_length)

        silent_regions = []
        start = None
        for i, is_silent in enumerate(silent_frames):
            sample = frame_times[min(i, len(frame_times) - 1)]
            if is_silent and start is None:
                start = sample
            elif not is_silent and start is not None:
                if sample - start >= min_silence_samples:
                    silent_regions.append((start, sample))
                start = None
        if start is not None and total_samples - start >= min_silence_samples:
            silent_regions.append((start, total_samples))

        if not silent_regions:
            emit_log("✅ No long silences found", "success", sid=sid)
            with temp_audio_file(suffix='.wav') as temp_wav:
                sf.write(temp_wav, audio, sr)
                run_subprocess_with_timeout(['ffmpeg', '-i', temp_wav, '-b:a', '256k', output_path, '-y'], sid=sid)
            return True

        emit_log(f"✅ Found {len(silent_regions)} silent segments", "success", sid=sid)

        keep_segments = []
        last_end = 0
        for start, end in silent_regions:
            if start > last_end:
                keep_segments.append((last_end, start))
            last_end = end
        if last_end < total_samples:
            keep_segments.append((last_end, total_samples))

        keep_segments = [(max(0, s), min(total_samples, e)) for s, e in keep_segments if e > s]
        if not keep_segments:
            emit_log("⚠️ No valid segments found, keeping original", "warning", sid=sid)
            shutil.copy(audio_path, output_path)
            return True

        merged_segments = []
        for seg in keep_segments:
            if merged_segments and seg[0] - merged_segments[-1][1] < sr * 0.1:
                merged_segments[-1] = (merged_segments[-1][0], seg[1])
            else:
                merged_segments.append(seg)
        keep_segments = merged_segments
        emit_log(f"🧩 Keeping {len(keep_segments)} segments", "info", sid=sid)

        final_audio = []
        last_end = 0
        for i, (start, end) in enumerate(keep_segments):
            start_padded = max(last_end, start - keep_silence_samples)
            end_padded = min(total_samples, end + keep_silence_samples)
            emit_log(f"🧩 Segment {i + 1}: {start_padded / sr:.2f}s → {end_padded / sr:.2f}s", "info", sid=sid)
            segment = audio[start_padded:end_padded]
            final_audio.append(segment)
            last_end = end_padded

        if not final_audio:
            emit_log("⚠️ No audio segments to keep, using original", "warning", sid=sid)
            shutil.copy(audio_path, output_path)
            return True

        final_audio = np.concatenate(final_audio)
        emit_log("💾 Saving processed audio...", "info", sid=sid)

        with temp_audio_file(suffix='.wav') as temp_wav:
            sf.write(temp_wav, final_audio, sr)
            try:
                run_subprocess_with_timeout(['ffmpeg', '-i', temp_wav, '-b:a', '256k', output_path, '-y'], sid=sid)
            except AudioProcessingError as e:
                emit_log(f"⚠️ FFmpeg conversion failed: {str(e)}, keeping WAV", "warning", sid=sid)
                shutil.move(temp_wav, output_path)
                return True

        original_duration = total_samples / sr
        final_duration = len(final_audio) / sr
        emit_log(f"⏱️ Original: {original_duration:.1f}s → Final: {final_duration:.1f}s "
                 f"({(original_duration - final_duration) / original_duration * 100:.1f}% removed)", "success", sid=sid)

        return True

    except Exception as e:
        emit_log(f"❌ Fatal error: {str(e)}", "error", error_context=str(e), sid=sid)
        try:
            shutil.copy(audio_path, output_path)
            emit_log("🔄 Copied original file as fallback", "info", sid=sid)
            return True
        except Exception as copy_e:
            emit_log(f"❌ Fallback copy failed: {str(copy_e)}", "error", error_context=str(copy_e), sid=sid)
            return False


class ProcessForm(FlaskForm):
    link = StringField('YouTube URL', validators=[
        Optional(),
        Regexp(r'https?://(www\.)?(youtube\.com|youtu\.be)/', message="Invalid YouTube URL")
    ])
    audio_file = FileField('Upload Audio File', validators=[Optional()])
    remove_silence = BooleanField('Remove Silence', default=True)
    enhance_vocals = BooleanField('Enhance Vocals', default=True)
    submit = SubmitField('Remove Instruments')


# Instrument Removal and File Processing Logic
def process_files(selected_files, session_id, remove_silence_enabled, enhance_vocals_enabled, silence_thresh,
                  min_silence_len, keep_silence, sid):
    user_dirs = get_user_directories(session_id)
    processed_files = []
    total_files = len(selected_files)
    total_steps = 3 if remove_silence_enabled and enhance_vocals_enabled else 2 if remove_silence_enabled or enhance_vocals_enabled else 1

    emit_log(f"🎵 Starting vocal extraction from {total_files} file(s)...", "info", sid=sid)

    for i, filename in enumerate(selected_files, 1):
        emit_log(f"📁 Processing file {i}/{total_files}: {filename}", "info", sid=sid)
        input_path = os.path.join(user_dirs['download'], filename)
        step = 1

        if not os.path.exists(input_path):
            emit_log(f"❌ Input file not found: {filename}", "error", sid=sid)
            continue

        emit_log("🎤 Isolating vocals with AI model...", "info", sid=sid)
        emit_progress(i, total_files, step, total_steps, "Isolating Vocals", sid=sid)
        if shutil.which('demucs') is None:
            emit_log("❌ Demucs not found. Install it with 'pip install demucs'.", "error", sid=sid)
            continue

        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=progress_heartbeat, args=("Vocal isolation", stop_heartbeat, sid))
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

        try:
            demucs_cmd = ['demucs', input_path] if not enhance_vocals_enabled else ['demucs', '--two-stems=vocals',
                                                                                    '-o', user_dirs['separated'],
                                                                                    input_path]

            def demucs_progress(progress):
                emit_progress(i, total_files, step, total_steps, f"Isolating Vocals ({progress:.1f}%)", sid=sid)

            result = run_subprocess_with_timeout(demucs_cmd, progress_callback=demucs_progress, sid=sid)
            emit_log("✅ Vocal isolation completed!", "success", sid=sid)
        except AudioProcessingError as e:
            emit_log(f"❌ Vocal isolation failed: {str(e)}", "error", error_context=str(e), sid=sid)
            stop_heartbeat.set()
            continue
        finally:
            stop_heartbeat.set()

        base_name = os.path.splitext(filename)[0]
        demucs_output_dir = os.path.join(user_dirs['separated'], 'htdemucs', base_name)
        vocals_file = os.path.join(demucs_output_dir, 'vocals.wav')

        if not os.path.exists(vocals_file):
            emit_log(f"❌ Could not find vocals file for {filename}", "error", sid=sid)
            try:
                demucs_dir_contents = os.listdir(demucs_output_dir) if os.path.exists(demucs_output_dir) else []
                emit_log(f"🔍 Demucs output directory contents: {demucs_dir_contents}", "info", sid=sid)
            except Exception as e:
                emit_log(f"⚠️ Failed to list Demucs output directory: {str(e)}", "warning", sid=sid)
            continue

        step += 1
        if enhance_vocals_enabled:
            emit_log("🎵 Enhancing vocal quality...", "info", sid=sid)
            emit_progress(i, total_files, step, total_steps, "Enhancing Vocals", sid=sid)
            stop_heartbeat = threading.Event()
            heartbeat_thread = threading.Thread(target=progress_heartbeat,
                                                args=("Vocal enhancement", stop_heartbeat, sid))
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            try:
                vocals, sr = librosa.load(vocals_file, sr=None)
                vocals = enhance_vocals(vocals, sr)
                enhanced_vocals_file = os.path.join(demucs_output_dir, f"{base_name}_enhanced_vocals.wav")
                sf.write(enhanced_vocals_file, vocals, sr)
                vocals_file = enhanced_vocals_file
                emit_log("✅ Vocal enhancement completed!", "success", sid=sid)
            except Exception as e:
                emit_log(f"⚠️ Enhancement failed, using original: {str(e)}", "warning", sid=sid)
                stop_heartbeat.set()
                continue
            finally:
                stop_heartbeat.set()
            step += 1

        final_filename = f"{base_name}_vocals_only.mp3"
        final_path = os.path.join(user_dirs['final'], final_filename)

        if remove_silence_enabled:
            emit_log("🔇 Removing silence from vocals...", "info", sid=sid)
            emit_progress(i, total_files, step, total_steps, "Removing Silence", sid=sid)
            stop_heartbeat = threading.Event()
            heartbeat_thread = threading.Thread(target=progress_heartbeat,
                                                args=("Silence removal", stop_heartbeat, sid))
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            try:
                success = remove_silence(
                    vocals_file,
                    final_path,
                    silence_thresh=silence_thresh,
                    min_silence_len=min_silence_len,
                    keep_silence=keep_silence,
                    sid=sid
                )
                if success:
                    processed_files.append(final_filename)
                    emit_log(f"🎉 Completed: {final_filename}", "success", sid=sid)
                else:
                    emit_log("⚠️ Silence removal failed, converting vocals to MP3...", "warning", sid=sid)
                    try:
                        run_subprocess_with_timeout(['ffmpeg', '-i', vocals_file, '-b:a', '256k', final_path, '-y'],
                                                    sid=sid)
                        processed_files.append(final_filename)
                        emit_log(f"🎉 Fallback conversion completed: {final_filename}", "success", sid=sid)
                    except AudioProcessingError as e:
                        emit_log(f"❌ Fallback conversion failed: {str(e)}", "error", error_context=str(e), sid=sid)
                        continue
            finally:
                stop_heartbeat.set()
        else:
            emit_log("💾 Converting vocals to MP3...", "info", sid=sid)
            emit_progress(i, total_files, step, total_steps, "Converting to MP3", sid=sid)
            try:
                audio, sr = librosa.load(vocals_file, sr=None)
                with temp_audio_file(suffix='.wav') as temp_wav:
                    sf.write(temp_wav, audio, sr)
                    run_subprocess_with_timeout(['ffmpeg', '-i', temp_wav, '-b:a', '256k', final_path, '-y'], sid=sid)
                processed_files.append(final_filename)
                emit_log(f"🎉 Completed: {final_filename}", "success", sid=sid)
            except AudioProcessingError as e:
                emit_log(f"❌ Conversion failed: {str(e)}", "error", error_context=str(e), sid=sid)
                continue

    socketio.emit('processing_complete', {'files': processed_files}, room=sid)
    return processed_files


@app.route('/final_output/<filename>')
def serve_final_output(filename):
    session_id = session.get('session_id')
    if not session_id:
        emit_log("❌ No session ID found", "error", sid=session_id)
        abort(403)
    try:
        filename = sanitize_filename(filename)
        user_final_dir = get_user_directories(session_id)['final']
        file_path = os.path.join(user_final_dir, filename)
        if not os.path.exists(file_path):
            emit_log(f"❌ File not found: {filename}", "error", sid=session_id)
            abort(404)
        return send_from_directory(user_final_dir, filename)
    except ValidationError as e:
        emit_log(f"❌ Invalid filename: {str(e)}", "error", sid=session_id)
        abort(400)


@app.route('/download-all', methods=['GET'])
def download_all_files():
    session_id = session.get('session_id')
    if not session_id:
        emit_log("❌ No session ID found", "error", sid=session_id)
        abort(403, description="No session ID found")

    file_names_json = request.args.get('files')
    if not file_names_json:
        emit_log("❌ No files specified for download", "error", sid=session_id)
        abort(400, description="No files specified for download")

    try:
        file_names = json.loads(file_names_json)
        if not isinstance(file_names, list) or not file_names:
            emit_log("❌ Invalid or empty file list format", "error", sid=session_id)
            abort(400, description="Invalid or empty file list format")
    except json.JSONDecodeError as e:
        emit_log(f"❌ Invalid JSON format for files parameter: {str(e)}", "error", error_context=str(e), sid=session_id)
        abort(400, description="Invalid JSON format for files parameter")

    user_final_dir = get_user_directories(session_id)['final']
    zip_base_name = "JustVocals_Extracted_Audio"
    temp_content_dir = None
    temp_zip_output_dir = None

    try:
        emit_log("📦 Preparing to zip files...", "info", sid=session_id)
        temp_content_dir = tempfile.mkdtemp()
        temp_zip_output_dir = tempfile.mkdtemp()
        valid_files = []

        for fname in file_names:
            try:
                sanitized_fname = sanitize_filename(fname)
                src_path = os.path.join(user_final_dir, sanitized_fname)
                if not os.path.exists(src_path):
                    emit_log(f"⚠️ File not found for zipping: {sanitized_fname}", "warning", sid=session_id)
                    continue
                if not os.access(src_path, os.R_OK):
                    emit_log(f"⚠️ File not readable: {sanitized_fname}", "warning", sid=session_id)
                    continue
                dest_path = os.path.join(temp_content_dir, sanitized_fname)
                shutil.copy2(src_path, dest_path)
                valid_files.append(sanitized_fname)
                emit_log(f"✅ Added {sanitized_fname} to zip", "success", sid=session_id)
            except ValidationError as e:
                emit_log(f"⚠️ Invalid filename for zipping: {fname} - {str(e)}", "warning", sid=session_id)
                continue
            except OSError as e:
                emit_log(f"⚠️ Error copying file {fname}: {str(e)}", "warning", error_context=str(e), sid=session_id)
                continue

        if not valid_files:
            emit_log("❌ No valid files available to zip", "error", sid=session_id)
            abort(400, description="No valid files available to zip")

        zip_file_path = os.path.join(temp_zip_output_dir, zip_base_name)
        try:
            zip_file_path = shutil.make_archive(
                zip_file_path,
                'zip',
                root_dir=temp_content_dir,
                base_dir='.'
            )
            emit_log(f"✅ Zip file created: {os.path.basename(zip_file_path)}", "success", sid=session_id)
        except Exception as e:
            emit_log(f"❌ Failed to create zip file: {str(e)}", "error", error_context=str(e), sid=session_id)
            abort(500, description=f"Failed to create zip file: {str(e)}")

        try:
            return send_file(
                zip_file_path,
                as_attachment=True,
                download_name=f"{zip_base_name}.zip",
                mimetype='application/zip'
            )
        except Exception as e:
            emit_log(f"❌ Failed to send zip file: {str(e)}", "error", error_context=str(e), sid=session_id)
            abort(500, description=f"Failed to send zip file: {str(e)}")

    except Exception as e:
        emit_log(f"❌ Error during zip preparation: {str(e)}", "error", error_context=str(e), sid=session_id)
        abort(500, description=f"Error during zip preparation: {str(e)}")
    finally:
        for d in [temp_content_dir, temp_zip_output_dir]:
            if d and os.path.exists(d):
                try:
                    shutil.rmtree(d)
                    emit_log(f"🧹 Cleaned up temporary directory: {d}", "info", sid=session_id)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir {d}: {str(e)}")


@socketio.on('connect')
def handle_connect():
    session_id = session.get('session_id')
    if not session_id:
        socketio.emit('error', {'message': 'No session ID found. Please reload the page.'})
        logger.warning("Client connected without session ID")
        return
    join_room(session_id)
    logger.info(f"Client connected: {session_id}")
    emit_log("✅ Connected to server", "success", sid=session_id)


@socketio.on('process_form')
def handle_process_form(data):
    form = ProcessForm(data=data)
    session_id = session.get('session_id')
    if not session_id:
        emit_log("❌ No session ID found", "error", sid=session_id)
        socketio.emit('error', {'message': 'No session ID found'}, room=session_id)
        return

    if not form.link.data and not form.audio_file.data:
        emit_log("❌ Please provide either a YouTube URL or an audio file", "error", sid=session_id)
        socketio.emit('error', {'message': 'Please provide either a YouTube URL or an audio file'}, room=session_id)
        return

    if form.link.data and form.audio_file.data:
        emit_log("❌ Please provide either a YouTube URL or an audio file, not both", "error", sid=session_id)
        socketio.emit('error', {'message': 'Please provide either a YouTube URL or an audio file, not both'},
                      room=session_id)
        return

    if not form.validate():
        for field, errors in form.errors.items():
            for error in errors:
                emit_log(f"Validation Error in {field}: {error}", "error", sid=session_id)
                socketio.emit('error', {'message': f"Validation Error in {field}: {error}"}, room=session_id)
        return

    user_dirs = get_user_directories(session_id)
    for dir_type in user_dirs.values():
        os.makedirs(dir_type, exist_ok=True)

    remove_silence_enabled = form.remove_silence.data
    enhance_vocals_enabled = form.enhance_vocals.data
    silence_thresh = int(data.get('silence_thresh', Config.SILENCE_THRESH_DEFAULT))
    min_silence_len = int(data.get('min_silence_len', Config.MIN_SILENCE_LEN_DEFAULT))
    keep_silence = int(data.get('keep_silence', Config.KEEP_SILENCE_DEFAULT))

    def process_thread_target(downloaded_files):
        try:
            processed_result_filenames = process_files(
                downloaded_files,
                session_id,
                remove_silence_enabled,
                enhance_vocals_enabled,
                silence_thresh,
                min_silence_len,
                keep_silence,
                sid=session_id
            )
            socketio.emit('processing_complete', {'files': processed_result_filenames}, room=session_id)
        except Exception as e:
            emit_log(f"❌ Processing failed: {str(e)}", "error", error_context=str(e), sid=session_id)
            socketio.emit('error', {'message': f"Processing failed: {str(e)}"}, room=session_id)

    if form.audio_file.data:
        file = form.audio_file.data
        filename = file.filename
        try:
            filename = sanitize_filename(filename)
            file_path = os.path.join(user_dirs['download'], filename)
            os.makedirs(user_dirs['download'], exist_ok=True)
            file.save(file_path)
            validate_audio_file_fast(file_path)
            emit_log(f"✅ Uploaded file: {filename}", "success", sid=session_id)
            thread = threading.Thread(target=process_thread_target, args=([filename],))
            thread.daemon = True
            thread.start()
            socketio.emit('processing_started', {}, room=session_id)
        except ValidationError as e:
            emit_log(f"❌ Invalid file: {str(e)}", "error", sid=session_id)
            socketio.emit('error', {'message': f"Invalid file: {str(e)}"}, room=session_id)
        except Exception as e:
            emit_log(f"❌ File upload failed: {str(e)}", "error", error_context=str(e), sid=session_id)
            socketio.emit('error', {'message': f"File upload failed: {str(e)}"}, room=session_id)
        return

    url = form.link.data

    def download_and_process_thread_target():
        with download_lock:
            if url in active_downloads:
                emit_log(f"ℹ️ Download for {url} already in progress", "info", sid=session_id)
                socketio.emit('error', {'message': f"Download for {url} already in progress"}, room=session_id)
                return
            active_downloads.add(url)

        try:
            emit_log("🚀 Starting YouTube download...", "info", sid=session_id)
            last_progress_time = time.time()

            def progress_hook(d):
                nonlocal last_progress_time
                if d['status'] == 'downloading':
                    current_time = time.time()
                    if current_time - last_progress_time >= Config.LOG_UPDATE_INTERVAL:
                        emit_log(f"⬇️ Download Progress: {d.get('_percent_str', 'N/A')}", "info", sid=session_id)
                        last_progress_time = current_time
                elif d['status'] == 'finished':
                    emit_log(f"✅ Finished downloading: {d.get('info_dict', {}).get('title', 'unknown')}", "success",
                             sid=session_id)

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(user_dirs['download'], '%(title)s_%(id)s.%(ext)s'),
                'restrictfilenames': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '256',
                }],
                'quiet': True,
                'no_warnings': True,
                'progress_hooks': [progress_hook],
                'noplaylist': False,
            }

            stop_heartbeat = threading.Event()
            heartbeat_thread = threading.Thread(target=progress_heartbeat,
                                                args=("YouTube download", stop_heartbeat, session_id))
            heartbeat_thread.daemon = True
            heartbeat_thread.start()

            try:
                initial_files = set(os.listdir(user_dirs['download']))
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
                    downloaded_files = []
                    if 'entries' in info_dict:
                        emit_log(f"📥 Detected playlist with {len(info_dict['entries'])} videos", "info", sid=session_id)
                        for entry in info_dict['entries']:
                            if entry:
                                filename = os.path.splitext(ydl.prepare_filename(entry))[0] + '.mp3'
                                if os.path.exists(filename):
                                    downloaded_files.append(os.path.basename(filename))
                    else:
                        filename = os.path.splitext(ydl.prepare_filename(info_dict))[0] + '.mp3'
                        if os.path.exists(filename):
                            downloaded_files.append(os.path.basename(filename))

                    final_files = set(os.listdir(user_dirs['download']))
                    downloaded_files = [f for f in final_files - initial_files if f.endswith('.mp3')]
                    if not downloaded_files:
                        raise DownloadError("No files were downloaded")
                    emit_log(
                        f"✅ Downloaded {len(downloaded_files)} file(s) successfully: {', '.join(downloaded_files)}",
                        "success", sid=session_id)
                    thread = threading.Thread(target=process_thread_target, args=(downloaded_files,))
                    thread.daemon = True
                    thread.start()
                    socketio.emit('processing_started', {}, room=session_id)
            finally:
                stop_heartbeat.set()

        except DownloadError as e:
            emit_log(f"❌ Download failed: {str(e)}", "error", error_context=str(e), sid=session_id)
            socketio.emit('error', {'message': f"Download failed: {str(e)}"}, room=session_id)
        except Exception as e:
            emit_log(f"❌ General download error: {str(e)}", "error", error_context=str(e), sid=session_id)
            socketio.emit('error', {'message': f"Download failed: {str(e)}"}, room=session_id)
        finally:
            with download_lock:
                active_downloads.discard(url)
                stop_heartbeat.set()

    thread = threading.Thread(target=download_and_process_thread_target)
    thread.daemon = True
    thread.start()
    socketio.emit('processing_started', {}, room=session_id)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Generate a new session ID for each request
    session['session_id'] = str(uuid.uuid4())
    session.permanent = True
    logger.info(f"New session created: {session['session_id']}")

    form = ProcessForm()
    processed_files = []

    processed_param = request.args.get('files')
    if processed_param:
        try:
            processed_files = json.loads(processed_param)
            if not isinstance(processed_files, list):
                processed_files = []
                emit_log("⚠️ Invalid processed files data", "warning", sid=session['session_id'])
        except json.JSONDecodeError:
            processed_files = []
            emit_log("⚠️ Failed to parse processed files data", "warning", sid=session['session_id'])

    return render_template('index.html', form=form, processed_files=processed_files, session_id=session['session_id'])


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