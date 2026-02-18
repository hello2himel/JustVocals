import os
import shutil
import threading
import json
import uuid
import time
import tempfile
import logging
from collections import deque

import yt_dlp
from flask import render_template, request, send_from_directory, session, abort, send_file
from flask_socketio import emit, join_room

from config import Config
from forms import ProcessForm
from utils import (
    get_user_directories, sanitize_filename, validate_audio_file_fast,
    emit_log, progress_heartbeat, ValidationError, DownloadError,
)
from audio_processing import process_files


logger = logging.getLogger(__name__)

# Active downloads tracking
active_downloads = set()
download_lock = threading.Lock()

# Per-session queue state: { session_id: { 'queue': deque, 'processing': bool, 'lock': Lock } }
session_queues = {}
session_queues_lock = threading.Lock()


def _get_session_queue(session_id):
    """Get or create queue state for a session."""
    with session_queues_lock:
        if session_id not in session_queues:
            session_queues[session_id] = {
                'queue': deque(),
                'processing': False,
                'lock': threading.Lock(),
            }
        return session_queues[session_id]


def _process_queue(socketio, session_id):
    """Process items from the session queue sequentially."""
    sq = _get_session_queue(session_id)

    while True:
        with sq['lock']:
            if not sq['queue']:
                sq['processing'] = False
                return
            job = sq['queue'].popleft()

        # Notify frontend about queue update
        _emit_queue_status(socketio, session_id, sq)

        emit_log(socketio, f"🎵 Starting next queued job...", "info", sid=session_id)
        socketio.emit('processing_started', {}, room=session_id)

        try:
            processed_result_filenames = process_files(
                socketio,
                job['files'],
                session_id,
                job['remove_silence'],
                job['enhance_vocals'],
                job['silence_thresh'],
                job['min_silence_len'],
                job['keep_silence'],
                sid=session_id
            )
            socketio.emit('processing_complete', {'files': processed_result_filenames}, room=session_id)
        except Exception as e:
            emit_log(socketio, f"❌ Processing failed: {str(e)}", "error", error_context=str(e), sid=session_id)
            socketio.emit('error', {'message': f"Processing failed: {str(e)}"}, room=session_id)

        # Notify frontend about queue update after job completes
        _emit_queue_status(socketio, session_id, sq)


def _emit_queue_status(socketio, session_id, sq):
    """Emit current queue status to frontend."""
    with sq['lock']:
        queue_items = [{'name': job.get('display_name', 'Unknown')} for job in sq['queue']]
        socketio.emit('queue_update', {
            'queue': queue_items,
            'processing': sq['processing'],
        }, room=session_id)


def _enqueue_job(socketio, session_id, job):
    """Add a job to the queue and start processing if not already running."""
    sq = _get_session_queue(session_id)
    with sq['lock']:
        sq['queue'].append(job)
        should_start = not sq['processing']
        if should_start:
            sq['processing'] = True

    _emit_queue_status(socketio, session_id, sq)

    if should_start:
        thread = threading.Thread(target=_process_queue, args=(socketio, session_id))
        thread.daemon = True
        thread.start()
    else:
        emit_log(socketio, f"📋 Added to queue (position {len(sq['queue'])})", "info", sid=session_id)
        socketio.emit('queued', {'position': len(sq['queue'])}, room=session_id)


def register_routes(app, socketio):
    @app.route('/final_output/<filename>')
    def serve_final_output(filename):
        session_id = session.get('session_id')
        if not session_id:
            emit_log(socketio, "❌ No session ID found", "error", sid=session_id)
            abort(403)
        try:
            filename = sanitize_filename(filename)
            user_final_dir = get_user_directories(session_id)['final']
            file_path = os.path.join(user_final_dir, filename)
            if not os.path.exists(file_path):
                emit_log(socketio, f"❌ File not found: {filename}", "error", sid=session_id)
                abort(404)
            return send_from_directory(user_final_dir, filename)
        except ValidationError as e:
            emit_log(socketio, f"❌ Invalid filename: {str(e)}", "error", sid=session_id)
            abort(400)

    @app.route('/download-all', methods=['GET'])
    def download_all_files():
        session_id = session.get('session_id')
        if not session_id:
            emit_log(socketio, "❌ No session ID found", "error", sid=session_id)
            abort(403, description="No session ID found")

        file_names_json = request.args.get('files')
        if not file_names_json:
            emit_log(socketio, "❌ No files specified for download", "error", sid=session_id)
            abort(400, description="No files specified for download")

        try:
            file_names = json.loads(file_names_json)
            if not isinstance(file_names, list) or not file_names:
                emit_log(socketio, "❌ Invalid or empty file list format", "error", sid=session_id)
                abort(400, description="Invalid or empty file list format")
        except json.JSONDecodeError as e:
            emit_log(socketio, f"❌ Invalid JSON format for files parameter: {str(e)}", "error", error_context=str(e), sid=session_id)
            abort(400, description="Invalid JSON format for files parameter")

        user_final_dir = get_user_directories(session_id)['final']
        zip_base_name = "JustVocals_Extracted_Audio"
        temp_content_dir = None
        temp_zip_output_dir = None

        try:
            emit_log(socketio, "📦 Preparing to zip files...", "info", sid=session_id)
            temp_content_dir = tempfile.mkdtemp()
            temp_zip_output_dir = tempfile.mkdtemp()
            valid_files = []

            for fname in file_names:
                try:
                    sanitized_fname = sanitize_filename(fname)
                    src_path = os.path.join(user_final_dir, sanitized_fname)
                    if not os.path.exists(src_path):
                        emit_log(socketio, f"⚠️ File not found for zipping: {sanitized_fname}", "warning", sid=session_id)
                        continue
                    if not os.access(src_path, os.R_OK):
                        emit_log(socketio, f"⚠️ File not readable: {sanitized_fname}", "warning", sid=session_id)
                        continue
                    dest_path = os.path.join(temp_content_dir, sanitized_fname)
                    shutil.copy2(src_path, dest_path)
                    valid_files.append(sanitized_fname)
                    emit_log(socketio, f"✅ Added {sanitized_fname} to zip", "success", sid=session_id)
                except ValidationError as e:
                    emit_log(socketio, f"⚠️ Invalid filename for zipping: {fname} - {str(e)}", "warning", sid=session_id)
                    continue
                except OSError as e:
                    emit_log(socketio, f"⚠️ Error copying file {fname}: {str(e)}", "warning", error_context=str(e), sid=session_id)
                    continue

            if not valid_files:
                emit_log(socketio, "❌ No valid files available to zip", "error", sid=session_id)
                abort(400, description="No valid files available to zip")

            zip_file_path = os.path.join(temp_zip_output_dir, zip_base_name)
            try:
                zip_file_path = shutil.make_archive(
                    zip_file_path,
                    'zip',
                    root_dir=temp_content_dir,
                    base_dir='.'
                )
                emit_log(socketio, f"✅ Zip file created: {os.path.basename(zip_file_path)}", "success", sid=session_id)
            except Exception as e:
                emit_log(socketio, f"❌ Failed to create zip file: {str(e)}", "error", error_context=str(e), sid=session_id)
                abort(500, description=f"Failed to create zip file: {str(e)}")

            try:
                return send_file(
                    zip_file_path,
                    as_attachment=True,
                    download_name=f"{zip_base_name}.zip",
                    mimetype='application/zip'
                )
            except Exception as e:
                emit_log(socketio, f"❌ Failed to send zip file: {str(e)}", "error", error_context=str(e), sid=session_id)
                abort(500, description=f"Failed to send zip file: {str(e)}")

        except Exception as e:
            emit_log(socketio, f"❌ Error during zip preparation: {str(e)}", "error", error_context=str(e), sid=session_id)
            abort(500, description=f"Error during zip preparation: {str(e)}")
        finally:
            for d in [temp_content_dir, temp_zip_output_dir]:
                if d and os.path.exists(d):
                    try:
                        shutil.rmtree(d)
                        emit_log(socketio, f"🧹 Cleaned up temporary directory: {d}", "info", sid=session_id)
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
        emit_log(socketio, "✅ Connected to server", "success", sid=session_id)

    @socketio.on('get_queue_status')
    def handle_get_queue_status():
        session_id = session.get('session_id')
        if not session_id:
            return
        sq = _get_session_queue(session_id)
        _emit_queue_status(socketio, session_id, sq)

    @socketio.on('process_form')
    def handle_process_form(data):
        form = ProcessForm(data=data)
        session_id = session.get('session_id')
        if not session_id:
            emit_log(socketio, "❌ No session ID found", "error", sid=session_id)
            socketio.emit('error', {'message': 'No session ID found'}, room=session_id)
            return

        if not form.link.data and not form.audio_file.data:
            emit_log(socketio, "❌ Please provide either a YouTube URL or an audio file", "error", sid=session_id)
            socketio.emit('error', {'message': 'Please provide either a YouTube URL or an audio file'}, room=session_id)
            return

        if form.link.data and form.audio_file.data:
            emit_log(socketio, "❌ Please provide either a YouTube URL or an audio file, not both", "error", sid=session_id)
            socketio.emit('error', {'message': 'Please provide either a YouTube URL or an audio file, not both'},
                          room=session_id)
            return

        if not form.validate():
            for field, errors in form.errors.items():
                for error in errors:
                    emit_log(socketio, f"Validation Error in {field}: {error}", "error", sid=session_id)
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

        if form.audio_file.data:
            file = form.audio_file.data
            filename = file.filename
            try:
                filename = sanitize_filename(filename)
                file_path = os.path.join(user_dirs['download'], filename)
                os.makedirs(user_dirs['download'], exist_ok=True)
                file.save(file_path)
                validate_audio_file_fast(file_path)
                emit_log(socketio, f"✅ Uploaded file: {filename}", "success", sid=session_id)

                job = {
                    'files': [filename],
                    'remove_silence': remove_silence_enabled,
                    'enhance_vocals': enhance_vocals_enabled,
                    'silence_thresh': silence_thresh,
                    'min_silence_len': min_silence_len,
                    'keep_silence': keep_silence,
                    'display_name': filename,
                }
                _enqueue_job(socketio, session_id, job)
                socketio.emit('processing_started', {}, room=session_id)
            except ValidationError as e:
                emit_log(socketio, f"❌ Invalid file: {str(e)}", "error", sid=session_id)
                socketio.emit('error', {'message': f"Invalid file: {str(e)}"}, room=session_id)
            except Exception as e:
                emit_log(socketio, f"❌ File upload failed: {str(e)}", "error", error_context=str(e), sid=session_id)
                socketio.emit('error', {'message': f"File upload failed: {str(e)}"}, room=session_id)
            return

        url = form.link.data

        def download_and_enqueue():
            with download_lock:
                if url in active_downloads:
                    emit_log(socketio, f"ℹ️ Download for {url} already in progress", "info", sid=session_id)
                    socketio.emit('error', {'message': f"Download for {url} already in progress"}, room=session_id)
                    return
                active_downloads.add(url)

            try:
                emit_log(socketio, "🚀 Starting YouTube download...", "info", sid=session_id)
                last_progress_time = time.time()

                def progress_hook(d):
                    nonlocal last_progress_time
                    if d['status'] == 'downloading':
                        current_time = time.time()
                        if current_time - last_progress_time >= Config.LOG_UPDATE_INTERVAL:
                            emit_log(socketio, f"⬇️ Download Progress: {d.get('_percent_str', 'N/A')}", "info", sid=session_id)
                            last_progress_time = current_time
                    elif d['status'] == 'finished':
                        emit_log(socketio, f"✅ Finished downloading: {d.get('info_dict', {}).get('title', 'unknown')}", "success",
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
                                                    args=(socketio, "YouTube download", stop_heartbeat, session_id))
                heartbeat_thread.daemon = True
                heartbeat_thread.start()

                try:
                    initial_files = set(os.listdir(user_dirs['download']))
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(url, download=True)
                        downloaded_files = []
                        if 'entries' in info_dict:
                            emit_log(socketio, f"📥 Detected playlist with {len(info_dict['entries'])} videos", "info", sid=session_id)
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
                        emit_log(socketio,
                                 f"✅ Downloaded {len(downloaded_files)} file(s) successfully: {', '.join(downloaded_files)}",
                                 "success", sid=session_id)

                        display_name = info_dict.get('title', url)
                        job = {
                            'files': downloaded_files,
                            'remove_silence': remove_silence_enabled,
                            'enhance_vocals': enhance_vocals_enabled,
                            'silence_thresh': silence_thresh,
                            'min_silence_len': min_silence_len,
                            'keep_silence': keep_silence,
                            'display_name': display_name,
                        }
                        _enqueue_job(socketio, session_id, job)
                        socketio.emit('processing_started', {}, room=session_id)
                finally:
                    stop_heartbeat.set()

            except DownloadError as e:
                emit_log(socketio, f"❌ Download failed: {str(e)}", "error", error_context=str(e), sid=session_id)
                socketio.emit('error', {'message': f"Download failed: {str(e)}"}, room=session_id)
            except Exception as e:
                emit_log(socketio, f"❌ General download error: {str(e)}", "error", error_context=str(e), sid=session_id)
                socketio.emit('error', {'message': f"Download failed: {str(e)}"}, room=session_id)
            finally:
                with download_lock:
                    active_downloads.discard(url)
                    stop_heartbeat.set()

        thread = threading.Thread(target=download_and_enqueue)
        thread.daemon = True
        thread.start()
        socketio.emit('processing_started', {}, room=session_id)

    @app.route('/', methods=['GET', 'POST'])
    def index():
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
                    emit_log(socketio, "⚠️ Invalid processed files data", "warning", sid=session['session_id'])
            except json.JSONDecodeError:
                processed_files = []
                emit_log(socketio, "⚠️ Failed to parse processed files data", "warning", sid=session['session_id'])

        return render_template('index.html', form=form, processed_files=processed_files, session_id=session['session_id'])
