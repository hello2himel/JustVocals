import os
import shutil
import subprocess
import threading
import json
from flask import Flask, render_template, request, send_from_directory, Response, send_file
from flask_socketio import SocketIO, emit
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, FileField
from wtforms.validators import Regexp, Optional
import yt_dlp
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt
from concurrent.futures import ThreadPoolExecutor
import tempfile  # Import for temporary directory creation
import socket

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_EXTENSIONS'] = ['.mp3', '.wav']
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB upload limit
socketio = SocketIO(app, cors_allowed_origins="*")

DOWNLOAD_DIR = 'downloads'
OUTPUT_DIR = 'separated'
FINAL_DIR = 'final_output'

# Ensure folders exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Track active downloads to prevent duplicates
active_downloads = set()
download_lock = threading.Lock()

# Thread pool for subprocess tasks
executor = ThreadPoolExecutor(max_workers=2)


def emit_log(message, type="info"):
    """Emit log message to frontend"""
    socketio.emit('log_message', {'message': message, 'type': type})
    print(message)


def emit_progress(file_index, total_files, step, total_steps, stage_name="Processing"):
    """Emit progress update to frontend"""
    # Calculate progress for the current file within its steps
    progress_per_file = 100 / total_files
    progress_per_step_in_file = progress_per_file / total_steps
    current_file_base_progress = (file_index - 1) * progress_per_file
    current_step_progress = step * progress_per_step_in_file

    total_progress = current_file_base_progress + current_step_progress
    socketio.emit('progress_update', {'progress': round(total_progress, 1), 'stage': stage_name})


def remove_silence(audio_path, output_path, silence_thresh=-40, min_silence_len=2000, keep_silence=500):
    """Remove long silent segments from audio with simplified, robust processing"""
    filename = os.path.basename(audio_path)
    emit_log(f"🔇 Processing silence removal for {filename}...", "info")

    try:
        # Validate input file
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        if file_size > 100:
            emit_log(f"⚠️ Large file ({file_size:.1f}MB), processing may take time", "warning")

        # Load audio once
        audio, sr = librosa.load(audio_path, sr=None)
        total_samples = len(audio)
        emit_log(f"📈 Loaded audio: {total_samples / sr:.1f}s, {sr}Hz", "info")

        if sr < 8000 or sr > 192000:
            emit_log(f"⚠️ Unusual sample rate: {sr}Hz", "warning")

        # Convert thresholds
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(frame_length // 2)
        min_silence_samples = int(min_silence_len * sr / 1000)
        keep_silence_samples = int(keep_silence * sr / 1000)

        # Calculate RMS
        emit_log("🔄 Computing audio energy...", "info")
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        max_rms = np.max(rms)

        if max_rms == 0:
            emit_log("⚠️ Audio is completely silent, keeping original", "warning")
            shutil.copy(audio_path, output_path)
            return True

        # Dynamic threshold
        silence_thresh_linear = 10 ** (silence_thresh / 20)
        dynamic_thresh = max(silence_thresh_linear, max_rms * 0.05)  # Minimum 5% of max RMS
        emit_log(f"🔍 Using silence threshold: {20 * np.log10(dynamic_thresh):.1f}dB", "info")

        # Identify silent frames
        silent_frames = rms < dynamic_thresh
        if len(silent_frames) == 0:
            emit_log("⚠️ No frames detected, keeping original", "warning")
            shutil.copy(audio_path, output_path)
            return True

        # Convert frames to samples
        frame_times = librosa.frames_to_samples(np.arange(len(silent_frames)), hop_length=hop_length)

        # Find silent regions
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
            emit_log("✅ No long silences found", "success")
            temp_wav = output_path.replace('.mp3', '.wav')
            sf.write(temp_wav, audio, sr)
            subprocess.run(['ffmpeg', '-i', temp_wav, '-b:a', '256k', output_path, '-y'],
                           capture_output=True, check=True)
            os.remove(temp_wav)
            return True

        emit_log(f"✅ Found {len(silent_regions)} silent segments", "success")

        # Create keep segments
        keep_segments = []
        last_end = 0
        for start, end in silent_regions:
            if start > last_end:
                keep_segments.append((last_end, start))
            last_end = end
        if last_end < total_samples:
            keep_segments.append((last_end, total_samples))

        # Validate segments
        keep_segments = [(max(0, s), min(total_samples, e)) for s, e in keep_segments if e > s]
        if not keep_segments:
            emit_log("⚠️ No valid segments, keeping original", "warning")
            shutil.copy(audio_path, output_path)
            return True

        # Merge close segments
        merged_segments = []
        for seg in keep_segments:
            if merged_segments and seg[0] - merged_segments[-1][1] < sr * 0.1:
                merged_segments[-1] = (merged_segments[-1][0], seg[1])
            else:
                merged_segments.append(seg)
        keep_segments = merged_segments
        emit_log(f"🧩 Keeping {len(keep_segments)} segments", "info")

        # Extract segments with padding
        final_audio = []
        last_end = 0
        for i, (start, end) in enumerate(keep_segments):
            start_padded = max(last_end, start - keep_silence_samples)
            end_padded = min(total_samples, end + keep_silence_samples)
            emit_log(f"🧩 Segment {i + 1}: {start_padded / sr:.2f}s → {end_padded / sr:.2f}s", "info")
            segment = audio[start_padded:end_padded]
            final_audio.append(segment)
            last_end = end_padded

        if not final_audio:
            emit_log("⚠️ No audio segments to keep, using original", "warning")
            shutil.copy(audio_path, output_path)
            return True

        final_audio = np.concatenate(final_audio)
        emit_log("💾 Saving processed audio...", "info")

        # Save output
        temp_wav = output_path.replace('.mp3', '_temp.wav')
        sf.write(temp_wav, final_audio, sr)

        try:
            subprocess.run(['ffmpeg', '-i', temp_wav, '-b:a', '256k', output_path, '-y'],
                           capture_output=True, check=True)
            os.remove(temp_wav)
        except Exception as e:
            emit_log(f"⚠️ FFmpeg conversion failed: {str(e)}, keeping WAV", "warning")
            os.rename(temp_wav, output_path)

        original_duration = total_samples / sr
        final_duration = len(final_audio) / sr
        emit_log(f"⏱️ Original: {original_duration:.1f}s → Final: {final_duration:.1f}s "
                 f"({(original_duration - final_duration) / original_duration * 100:.1f}% removed)", "success")

        return True

    except Exception as e:
        emit_log(f"❌ Fatal error: {str(e)}", "error")
        try:
            shutil.copy(audio_path, output_path)
            emit_log("🔄 Copied original file as fallback", "info")
            return True
        except Exception as copy_e:
            emit_log(f"❌ Fallback copy failed: {str(copy_e)}", "error")
            return False


def enhance_vocals(vocals, sr):
    """Apply audio enhancement to vocals"""
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


class ProcessForm(FlaskForm):
    link = StringField('YouTube URL', validators=[
        Optional(),
        Regexp(r'https?://(www\.)?(youtube\.com|youtu\.be)/', message="Invalid YouTube URL")
    ])
    audio_file = FileField('Upload Audio File', validators=[Optional()])
    remove_silence = BooleanField('Remove Silence', default=True)
    enhance_vocals = BooleanField('Enhance Vocals', default=True)
    submit = SubmitField('Remove Instruments')


def run_subprocess(command):
    """Run subprocess command in a thread-safe manner"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result
    except Exception as e:
        return e


def process_files(selected_files, remove_silence_enabled, enhance_vocals_enabled, silence_thresh, min_silence_len,
                  keep_silence):
    processed_files = []
    total_files = len(selected_files)

    # Calculate total steps for progress bar based on enabled options
    # Step 1: Demucs (always)
    # Step 2: Enhance (if enabled)
    # Step 3: Remove Silence / Convert to MP3 (always, but 'Remove Silence' is more intensive)
    num_sub_steps_per_file = 1  # Start with Demucs
    if enhance_vocals_enabled:
        num_sub_steps_per_file += 1
    if remove_silence_enabled:
        num_sub_steps_per_file += 1
    else:  # If silence removal is off, there's still an MP3 conversion step
        num_sub_steps_per_file += 1

    emit_log(f"🎵 Starting vocal extraction from {total_files} file(s)...", "info")

    for i, filename in enumerate(selected_files, 1):
        emit_log(f"📁 Processing file {i}/{total_files}: {filename}", "info")
        input_path = os.path.join(DOWNLOAD_DIR, filename)
        current_step_in_file = 1

        # Step 1: Extract vocals using Demucs
        emit_log("🎤 Isolating vocals with AI model...", "info")
        emit_progress(i, total_files, current_step_in_file, num_sub_steps_per_file, "Isolating Vocals")
        if shutil.which('demucs') is None:
            emit_log("❌ Demucs not found. Install it with pip or conda.", "error")
            continue

        try:
            # Use --two-stems=vocals to only get vocals and other, or no arg for htdemucs
            demucs_cmd = ['demucs', input_path]
            # Use a specific model if needed, e.g., ['demucs', '-n', 'htdemucs_6s', input_path]
            future = executor.submit(run_subprocess, demucs_cmd)
            result = future.result()
            if isinstance(result, subprocess.CalledProcessError):
                emit_log(f"❌ Demucs error: {result.stderr}", "error")
                continue
            emit_log("✅ Vocal isolation completed!", "success")
        except Exception as e:
            emit_log(f"❌ Vocal isolation failed: {str(e)}", "error")
            continue

        # Find the vocals file - Demucs creates folders under 'separated/MODEL_NAME/'
        base_name = os.path.splitext(filename)[0]
        # Assuming 'htdemucs' is the default model or specified one
        demucs_output_dir = os.path.join(OUTPUT_DIR, 'htdemucs', base_name)
        vocals_file = os.path.join(demucs_output_dir, 'vocals.wav')

        if not os.path.exists(vocals_file):
            emit_log(f"❌ Could not find vocals file for {filename} at {vocals_file}", "error")
            continue

        current_step_in_file += 1
        # Step 2: Enhance vocals if enabled
        if enhance_vocals_enabled:
            emit_log("🎵 Enhancing vocal quality...", "info")
            emit_progress(i, total_files, current_step_in_file, num_sub_steps_per_file, "Enhancing Vocals")
            try:
                vocals, sr = librosa.load(vocals_file, sr=None)
                vocals = enhance_vocals(vocals, sr)
                enhanced_vocals_file = os.path.join(demucs_output_dir, 'vocals_enhanced.wav')
                sf.write(enhanced_vocals_file, vocals, sr)
                vocals_file = enhanced_vocals_file
                emit_log("✅ Vocal enhancement completed!", "success")
            except Exception as e:
                emit_log(f"⚠️ Enhancement failed for {filename}, using original vocals: {str(e)}", "warning")
            current_step_in_file += 1

        # Step 3: Remove silence or convert to final MP3
        final_filename = f"{base_name}_vocals_only.mp3"
        final_path = os.path.join(FINAL_DIR, final_filename)

        if remove_silence_enabled:
            emit_log("🔇 Removing silence from vocals...", "info")
            emit_progress(i, total_files, current_step_in_file, num_sub_steps_per_file, "Removing Silence")
            success = remove_silence(
                vocals_file,
                final_path,
                silence_thresh=silence_thresh,
                min_silence_len=min_silence_len,
                keep_silence=keep_silence
            )
            if success:
                processed_files.append(final_filename)
                emit_log(f"🎉 Completed: {final_filename}", "success")
            else:
                # Fallback: Convert vocals_file to MP3
                emit_log("⚠️ Silence removal failed, converting vocals to MP3 as fallback...", "warning")
                if shutil.which('ffmpeg') is None:
                    emit_log("❌ FFmpeg not found. Please install it.", "error")
                    continue
                try:
                    # Always convert to MP3 for final output
                    future = executor.submit(run_subprocess,
                                             ['ffmpeg', '-i', vocals_file, '-b:a', '256k', final_path, '-y'])
                    result = future.result()
                    if isinstance(result, subprocess.CalledProcessError):
                        emit_log(f"❌ FFmpeg error during fallback conversion for {filename}: {result.stderr}", "error")
                        continue
                    processed_files.append(final_filename)
                    emit_log(f"🎉 Fallback conversion completed: {final_filename}", "success")
                except Exception as e:
                    emit_log(f"❌ Fallback conversion failed for {filename}: {str(e)}", "error")
                    continue
        else:
            emit_log("💾 Converting vocals to MP3...", "info")
            emit_progress(i, total_files, current_step_in_file, num_sub_steps_per_file, "Converting to MP3")
            if shutil.which('ffmpeg') is None:
                emit_log("❌ FFmpeg not found. Please install it.", "error")
                continue

            try:
                # Ensure the vocal file exists before attempting to convert
                if os.path.exists(vocals_file):
                    # Convert the isolated vocals (which might be WAV) to MP3
                    future = executor.submit(run_subprocess,
                                             ['ffmpeg', '-i', vocals_file, '-b:a', '256k', final_path, '-y'])
                    result = future.result()
                    if isinstance(result, subprocess.CalledProcessError):
                        emit_log(f"❌ FFmpeg error during conversion for {filename}: {result.stderr}", "error")
                        continue
                    processed_files.append(final_filename)
                    emit_log(f"🎉 Completed: {final_filename}", "success")
                else:
                    emit_log(f"❌ Vocals file not found for conversion for {filename}", "error")
            except Exception as e:
                emit_log(f"❌ Conversion failed for {filename}: {str(e)}", "error")
                continue

    return processed_files


@app.route('/final_output/<filename>')
def serve_final_output(filename):
    """Serve files from the final_output directory"""
    return send_from_directory(FINAL_DIR, filename)


@app.route('/download-all', methods=['GET'])
def download_all_files():
    """
    Endpoint to zip and download multiple processed audio files.
    Expects a 'files' query parameter containing a JSON string of filenames.
    """
    file_names_json = request.args.get('files')
    if not file_names_json:
        return "No files specified for download.", 400

    try:
        file_names = json.loads(file_names_json)
        if not isinstance(file_names, list):
            return "Invalid file list format.", 400
    except json.JSONDecodeError:
        return "Invalid JSON format for files parameter.", 400

    zip_base_name = "JustVocals_Extracted_Audio"
    zip_file_path = None  # Define outside try for finally block

    # Create a temporary directory for the files to be zipped
    temp_content_dir = None
    # Create a temporary directory for the actual zip file output
    temp_zip_output_dir = None

    try:
        temp_content_dir = tempfile.mkdtemp()
        temp_zip_output_dir = tempfile.mkdtemp()

        # Copy each specified file to the temporary content directory
        for fname in file_names:
            src_path = os.path.join(FINAL_DIR, fname)
            if not os.path.exists(src_path):
                # Log error but continue with available files
                emit_log(f"⚠️ File not found for zipping: {fname}", "warning")
                continue
            shutil.copy(src_path, temp_content_dir)

        # Create the zip archive:
        # The zip file will be created in temp_zip_output_dir.
        # The contents to be zipped are taken from temp_content_dir,
        # with '.' meaning the direct contents of temp_content_dir are archived.
        zip_file_path = shutil.make_archive(
            os.path.join(temp_zip_output_dir, zip_base_name),
            'zip',
            root_dir=temp_content_dir,
            base_dir='.'
        )

        # Send the zip file
        return send_file(zip_file_path, as_attachment=True, download_name=os.path.basename(zip_file_path))

    except Exception as e:
        emit_log(f"❌ Error zipping files for download: {str(e)}", "error")
        return "Error zipping files.", 500
    finally:
        # Clean up both temporary directories
        if temp_content_dir and os.path.exists(temp_content_dir):
            shutil.rmtree(temp_content_dir)
        if temp_zip_output_dir and os.path.exists(temp_zip_output_dir):
            shutil.rmtree(temp_zip_output_dir)  # This will also remove the zip file itself


@app.route('/', methods=['GET', 'POST'])
def index():
    form = ProcessForm()
    processed_files = []

    # Handle query parameters for initial state (e.g., after a redirect)
    processed_param = request.args.get('files')  # Changed from 'processed' to 'files' for consistency
    if processed_param:
        try:
            # Expecting a JSON array of filenames
            processed_data = json.loads(processed_param)
            processed_files = processed_data
            if not isinstance(processed_files, list):
                processed_files = []
                emit_log("⚠️ Invalid processed files data received from URL", "warning")
        except json.JSONDecode_error:
            processed_files = []
            emit_log("⚠️ Failed to parse processed files data from URL", "warning")

    if request.method == 'POST':
        # Validate that either link or file is provided, but not both
        link_provided = form.link.data and form.link.data.strip()
        file_provided = form.audio_file.data and hasattr(form.audio_file.data,
                                                         'filename') and form.audio_file.data.filename

        if not link_provided and not file_provided:
            emit_log("❌ Please provide either a YouTube URL or an audio file.", "error")
            return render_template('index.html', form=form, processed_files=processed_files)

        if link_provided and file_provided:
            emit_log("❌ Please provide either a YouTube URL or an audio file, not both.", "error")
            return render_template('index.html', form=form, processed_files=processed_files)

        # WTForms validation
        if not form.validate_on_submit():
            emit_log("❌ Form validation failed!", "error")
            # If form.validate_on_submit() fails, Flask-WTF usually populates form.errors
            for field, errors in form.errors.items():
                for error in errors:
                    emit_log(f"Validation Error in {field}: {error}", "error")
            return render_template('index.html', form=form, processed_files=processed_files)

        remove_silence_enabled = form.remove_silence.data
        enhance_vocals_enabled = form.enhance_vocals.data
        silence_thresh = int(request.form.get('silence_thresh', -40))
        min_silence_len = int(request.form.get('min_silence_len', 2000))  # Expected milliseconds for backend
        keep_silence = int(request.form.get('keep_silence', 500))

        def process_thread_target(downloaded_files):
            # This function will run in a separate thread
            processed_result_filenames = process_files(
                downloaded_files,
                remove_silence_enabled,
                enhance_vocals_enabled,
                silence_thresh,
                min_silence_len,
                keep_silence
            )
            # Emit the actual list of processed filenames when complete
            socketio.emit('processing_complete', {'files': processed_result_filenames})

        if file_provided:
            file = form.audio_file.data
            filename = file.filename
            # Basic extension check, more robust validation is in JS
            if filename and any(filename.lower().endswith(ext) for ext in app.config['UPLOAD_EXTENSIONS']):
                file_path = os.path.join(DOWNLOAD_DIR, filename)
                # Ensure the download directory exists before saving
                os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                file.save(file_path)
                emit_log(f"✅ Uploaded file: {filename}", "success")
                # Start processing in a new thread
                thread = threading.Thread(target=process_thread_target, args=([filename],))
                thread.daemon = True
                thread.start()
                return Response(status=204)  # Return 204 No Content for successful start of async process

            emit_log("❌ Invalid file format! Use .mp3 or .wav.", "error")
            return render_template('index.html', form=form, processed_files=processed_files)

        url = form.link.data

        def download_and_process_thread_target():
            with download_lock:
                if url in active_downloads:
                    emit_log(f"ℹ️ Download for {url} already in progress, skipping new download...", "info")
                    return  # Exit if already downloading
                active_downloads.add(url)

            downloaded_files = []
            try:
                emit_log("🚀 Starting YouTube download...", "info")
                emit_log(f"📥 Downloading from: {url}", "info")

                if shutil.which('ffmpeg') is None:
                    emit_log("❌ FFmpeg not found. Please install it for audio conversion.", "error")
                    socketio.emit('error', {'message': 'Server error: FFmpeg not found.'})
                    return

                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s_%(id)s.%(ext)s'),
                    'restrictfilenames': True,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '256',  # High quality MP3
                    }],
                    'quiet': True,  # Suppress console output from yt-dlp
                    'no_warnings': True,  # Suppress warnings from yt-dlp
                    'progress_hooks': [
                        lambda d: emit_log(f"Download Progress: {d['_percent_str'] or 'N/A'}", "info") if d[
                                                                                                              'status'] == 'downloading' else None],
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
                    # Get the actual filename after post-processing (e.g., mp3 conversion)
                    # This can be tricky, often it's the 'filepath' or constructed from 'title' and 'ext'
                    # For simplicity, we'll scan the DOWNLOAD_DIR for new mp3s
                    downloaded_filename_base = ydl.prepare_filename(info_dict)
                    # After post-processing to mp3, the name will be .mp3
                    downloaded_filename = os.path.splitext(downloaded_filename_base)[0] + '.mp3'

                    if os.path.exists(downloaded_filename):
                        downloaded_files.append(os.path.basename(downloaded_filename))
                        emit_log(f"✅ Downloaded {os.path.basename(downloaded_filename)} successfully!", "success")
                    else:
                        emit_log("❌ Downloaded file not found after yt-dlp operation.", "error")
                        socketio.emit('error', {'message': 'Failed to locate downloaded YouTube audio file.'})
                        return

            except yt_dlp.utils.DownloadError as e:
                emit_log(f"❌ YouTube Download Error: {e.exc_info[0].__name__}: {e.exc_info[1]}", "error")
                socketio.emit('error', {'message': f"YouTube download failed: {str(e)}"})
                return
            except Exception as e:
                emit_log(f"❌ General Download Failed: {str(e)}", "error")
                socketio.emit('error', {'message': f"Download failed: {str(e)}"})
                return
            finally:
                with download_lock:
                    if url in active_downloads:
                        active_downloads.remove(url)

            if downloaded_files:
                thread = threading.Thread(target=process_thread_target, args=(downloaded_files,))
                thread.daemon = True
                thread.start()

        # Start the download and processing in a new thread
        threading.Thread(target=download_and_process_thread_target).start()
        return Response(status=204)  # Return 204 No Content for successful start of async process

    return render_template('index.html', form=form, processed_files=processed_files)


if __name__ == '__main__':
    # Ensure all necessary directories exist at startup
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # Get local network IP
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to a public IP (Google DNS) to get local IP
        local_ip = s.getsockname()[0]
        s.close()
    except Exception as e:
        local_ip = "unknown"
        emit_log(f"⚠️ Could not determine local network IP: {str(e)}", "warning")

    # Log both URLs
    emit_log(f"Server starting...\n"
             f"This code is not yet suitable for production. Please consider contributing.\n"
             f"🌐 Localhost: http://localhost:5000\n"
             f"📡 Local Network: http://{local_ip}:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

