import os
import shutil
import threading
import logging

import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt

from utils import (
    get_user_directories, temp_audio_file, emit_log, emit_progress,
    progress_heartbeat, run_subprocess_with_timeout, AudioProcessingError,
)
from config import Config


logger = logging.getLogger(__name__)


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
        logger.warning(f"Enhancement failed, using original: {str(e)}")
        return vocals


def remove_silence(socketio, audio_path, output_path, silence_thresh, min_silence_len, keep_silence, sid=None):
    filename = os.path.basename(audio_path)
    emit_log(socketio, f"🔇 Processing silence removal for {filename}...", "info", sid=sid)

    try:
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        if file_size > 100:
            emit_log(socketio, f"⚠️ Large file ({file_size:.1f}MB), processing may take time", "warning", sid=sid)

        audio, sr = librosa.load(audio_path, sr=None)
        total_samples = len(audio)
        emit_log(socketio, f"📈 Loaded audio: {total_samples / sr:.1f}s, {sr}Hz", "info", sid=sid)

        if sr < 8000 or sr > 192000:
            emit_log(socketio, f"⚠️ Unusual sample rate: {sr}Hz", "warning", sid=sid)

        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(frame_length // 2)
        min_silence_samples = int(min_silence_len * sr / 1000)
        keep_silence_samples = int(keep_silence * sr / 1000)

        emit_log(socketio, "🔄 Computing audio energy...", "info", sid=sid)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        max_rms = np.max(rms)

        if max_rms == 0:
            emit_log(socketio, "⚠️ Audio is completely silent, keeping original", "warning", sid=sid)
            shutil.copy(audio_path, output_path)
            return True

        silence_thresh_linear = 10 ** (silence_thresh / 20)
        dynamic_thresh = max(silence_thresh_linear, max_rms * 0.05)
        emit_log(socketio, f"🔍 Using silence threshold: {20 * np.log10(dynamic_thresh):.1f}dB", "info", sid=sid)

        silent_frames = rms < dynamic_thresh
        if len(silent_frames) == 0:
            emit_log(socketio, "⚠️ No frames detected, keeping original", "warning", sid=sid)
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
            emit_log(socketio, "✅ No long silences found", "success", sid=sid)
            with temp_audio_file(suffix='.wav') as temp_wav:
                sf.write(temp_wav, audio, sr)
                run_subprocess_with_timeout(['ffmpeg', '-i', temp_wav, '-b:a', '256k', output_path, '-y'], sid=sid)
            return True

        emit_log(socketio, f"✅ Found {len(silent_regions)} silent segments", "success", sid=sid)

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
            emit_log(socketio, "⚠️ No valid segments found, keeping original", "warning", sid=sid)
            shutil.copy(audio_path, output_path)
            return True

        merged_segments = []
        for seg in keep_segments:
            if merged_segments and seg[0] - merged_segments[-1][1] < sr * 0.1:
                merged_segments[-1] = (merged_segments[-1][0], seg[1])
            else:
                merged_segments.append(seg)
        keep_segments = merged_segments
        emit_log(socketio, f"🧩 Keeping {len(keep_segments)} segments", "info", sid=sid)

        final_audio = []
        last_end = 0
        for i, (start, end) in enumerate(keep_segments):
            start_padded = max(last_end, start - keep_silence_samples)
            end_padded = min(total_samples, end + keep_silence_samples)
            emit_log(socketio, f"🧩 Segment {i + 1}: {start_padded / sr:.2f}s → {end_padded / sr:.2f}s", "info", sid=sid)
            segment = audio[start_padded:end_padded]
            final_audio.append(segment)
            last_end = end_padded

        if not final_audio:
            emit_log(socketio, "⚠️ No audio segments to keep, using original", "warning", sid=sid)
            shutil.copy(audio_path, output_path)
            return True

        final_audio = np.concatenate(final_audio)
        emit_log(socketio, "💾 Saving processed audio...", "info", sid=sid)

        with temp_audio_file(suffix='.wav') as temp_wav:
            sf.write(temp_wav, final_audio, sr)
            try:
                run_subprocess_with_timeout(['ffmpeg', '-i', temp_wav, '-b:a', '256k', output_path, '-y'], sid=sid)
            except AudioProcessingError as e:
                emit_log(socketio, f"⚠️ FFmpeg conversion failed: {str(e)}, keeping WAV", "warning", sid=sid)
                shutil.move(temp_wav, output_path)
                return True

        original_duration = total_samples / sr
        final_duration = len(final_audio) / sr
        emit_log(socketio, f"⏱️ Original: {original_duration:.1f}s → Final: {final_duration:.1f}s "
                 f"({(original_duration - final_duration) / original_duration * 100:.1f}% removed)", "success", sid=sid)

        return True

    except Exception as e:
        emit_log(socketio, f"❌ Fatal error: {str(e)}", "error", error_context=str(e), sid=sid)
        try:
            shutil.copy(audio_path, output_path)
            emit_log(socketio, "🔄 Copied original file as fallback", "info", sid=sid)
            return True
        except Exception as copy_e:
            emit_log(socketio, f"❌ Fallback copy failed: {str(copy_e)}", "error", error_context=str(copy_e), sid=sid)
            return False


def process_files(socketio, selected_files, session_id, remove_silence_enabled, enhance_vocals_enabled, silence_thresh,
                  min_silence_len, keep_silence, sid):
    user_dirs = get_user_directories(session_id)
    processed_files = []
    total_files = len(selected_files)
    total_steps = 3 if remove_silence_enabled and enhance_vocals_enabled else 2 if remove_silence_enabled or enhance_vocals_enabled else 1

    emit_log(socketio, f"🎵 Starting vocal extraction from {total_files} file(s)...", "info", sid=sid)

    for i, filename in enumerate(selected_files, 1):
        emit_log(socketio, f"📁 Processing file {i}/{total_files}: {filename}", "info", sid=sid)
        input_path = os.path.join(user_dirs['download'], filename)
        step = 1

        if not os.path.exists(input_path):
            emit_log(socketio, f"❌ Input file not found: {filename}", "error", sid=sid)
            continue

        emit_log(socketio, "🎤 Isolating vocals with AI model...", "info", sid=sid)
        emit_progress(socketio, i, total_files, step, total_steps, "Isolating Vocals", sid=sid)
        if shutil.which('demucs') is None:
            emit_log(socketio, "❌ Demucs not found. Install it with 'pip install demucs'.", "error", sid=sid)
            continue

        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=progress_heartbeat, args=(socketio, "Vocal isolation", stop_heartbeat, sid))
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

        try:
            demucs_cmd = ['demucs', input_path] if not enhance_vocals_enabled else ['demucs', '--two-stems=vocals',
                                                                                    '-o', user_dirs['separated'],
                                                                                    input_path]

            def demucs_progress(progress):
                emit_progress(socketio, i, total_files, step, total_steps, f"Isolating Vocals ({progress:.1f}%)", sid=sid)

            result = run_subprocess_with_timeout(demucs_cmd, progress_callback=demucs_progress, sid=sid)
            emit_log(socketio, "✅ Vocal isolation completed!", "success", sid=sid)
        except AudioProcessingError as e:
            emit_log(socketio, f"❌ Vocal isolation failed: {str(e)}", "error", error_context=str(e), sid=sid)
            stop_heartbeat.set()
            continue
        finally:
            stop_heartbeat.set()

        base_name = os.path.splitext(filename)[0]
        demucs_output_dir = os.path.join(user_dirs['separated'], 'htdemucs', base_name)
        vocals_file = os.path.join(demucs_output_dir, 'vocals.wav')

        if not os.path.exists(vocals_file):
            emit_log(socketio, f"❌ Could not find vocals file for {filename}", "error", sid=sid)
            try:
                demucs_dir_contents = os.listdir(demucs_output_dir) if os.path.exists(demucs_output_dir) else []
                emit_log(socketio, f"🔍 Demucs output directory contents: {demucs_dir_contents}", "info", sid=sid)
            except Exception as e:
                emit_log(socketio, f"⚠️ Failed to list Demucs output directory: {str(e)}", "warning", sid=sid)
            continue

        step += 1
        if enhance_vocals_enabled:
            emit_log(socketio, "🎵 Enhancing vocal quality...", "info", sid=sid)
            emit_progress(socketio, i, total_files, step, total_steps, "Enhancing Vocals", sid=sid)
            stop_heartbeat = threading.Event()
            heartbeat_thread = threading.Thread(target=progress_heartbeat,
                                                args=(socketio, "Vocal enhancement", stop_heartbeat, sid))
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            try:
                vocals, sr = librosa.load(vocals_file, sr=None)
                vocals = enhance_vocals(vocals, sr)
                enhanced_vocals_file = os.path.join(demucs_output_dir, f"{base_name}_enhanced_vocals.wav")
                sf.write(enhanced_vocals_file, vocals, sr)
                vocals_file = enhanced_vocals_file
                emit_log(socketio, "✅ Vocal enhancement completed!", "success", sid=sid)
            except Exception as e:
                emit_log(socketio, f"⚠️ Enhancement failed, using original: {str(e)}", "warning", sid=sid)
                stop_heartbeat.set()
                continue
            finally:
                stop_heartbeat.set()
            step += 1

        final_filename = f"{base_name}_vocals_only.mp3"
        final_path = os.path.join(user_dirs['final'], final_filename)

        if remove_silence_enabled:
            emit_log(socketio, "🔇 Removing silence from vocals...", "info", sid=sid)
            emit_progress(socketio, i, total_files, step, total_steps, "Removing Silence", sid=sid)
            stop_heartbeat = threading.Event()
            heartbeat_thread = threading.Thread(target=progress_heartbeat,
                                                args=(socketio, "Silence removal", stop_heartbeat, sid))
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            try:
                success = remove_silence(
                    socketio,
                    vocals_file,
                    final_path,
                    silence_thresh=silence_thresh,
                    min_silence_len=min_silence_len,
                    keep_silence=keep_silence,
                    sid=sid
                )
                if success:
                    processed_files.append(final_filename)
                    emit_log(socketio, f"🎉 Completed: {final_filename}", "success", sid=sid)
                else:
                    emit_log(socketio, "⚠️ Silence removal failed, converting vocals to MP3...", "warning", sid=sid)
                    try:
                        run_subprocess_with_timeout(['ffmpeg', '-i', vocals_file, '-b:a', '256k', final_path, '-y'],
                                                    sid=sid)
                        processed_files.append(final_filename)
                        emit_log(socketio, f"🎉 Fallback conversion completed: {final_filename}", "success", sid=sid)
                    except AudioProcessingError as e:
                        emit_log(socketio, f"❌ Fallback conversion failed: {str(e)}", "error", error_context=str(e), sid=sid)
                        continue
            finally:
                stop_heartbeat.set()
        else:
            emit_log(socketio, "💾 Converting vocals to MP3...", "info", sid=sid)
            emit_progress(socketio, i, total_files, step, total_steps, "Converting to MP3", sid=sid)
            try:
                audio, sr = librosa.load(vocals_file, sr=None)
                with temp_audio_file(suffix='.wav') as temp_wav:
                    sf.write(temp_wav, audio, sr)
                    run_subprocess_with_timeout(['ffmpeg', '-i', temp_wav, '-b:a', '256k', final_path, '-y'], sid=sid)
                processed_files.append(final_filename)
                emit_log(socketio, f"🎉 Completed: {final_filename}", "success", sid=sid)
            except AudioProcessingError as e:
                emit_log(socketio, f"❌ Conversion failed: {str(e)}", "error", error_context=str(e), sid=sid)
                continue

    return processed_files
