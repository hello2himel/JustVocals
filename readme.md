# JustVocals 🎤

JustVocals is a lightweight web app that lets you isolate clean vocals from YouTube videos or audio files using AI-powered tools like [Demucs](https://github.com/facebookresearch/demucs). With additional features like silence removal and vocal enhancement, it's a handy utility for musicians, producers, and remixers.

> 🧪 Built with Flask + SocketIO. Frontend is plain HTML/CSS/JS with zero frameworks.

---

## 🚀 Getting Started

### ⚙️ Requirements

- **Python 3.10 (recommended)**
- `ffmpeg`
- `demucs`
- Python dependencies: see `requirements.txt`

### 🔧 Setup

```bash
git clone https://github.com/hello2himel/justvocals.git
cd justvocals

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make sure ffmpeg and demucs are installed
which ffmpeg
which demucs
````

### ▶️ Run

```bash
python main.py
```

Then open your browser at: `http://127.0.0.1:5000`

---

## 🧠 How It Works

1. **Input Options**: User can upload an MP3/WAV or provide a YouTube link.
2. **Vocal Isolation**: Uses `demucs` with `--two-stems=vocals` to extract vocals.
3. **Optional Enhancements**:

   * Normalize volume and apply high-pass filtering
   * Soft compression to smooth dynamics
4. **Silence Removal**: Detects low-RMS regions and trims long silences (configurable).
5. **Output**: Returns MP3 files with isolated vocals, optionally enhanced and trimmed.

---

## 🛠️ TODO

* [ ] Make the codebase more production-ready (logging, error handling, etc.)
* [ ] Package and publish a Docker image
* [ ] Deploy on a public server (hosting costs required)

---

## 🙌 Support This Project

Hosting and development take time and funds. If you found JustVocals useful, please consider donating to help with server costs and feature development.

👉 [Donate here](https://hello2himel.netlify.app/donate?source=JustVocals&session_id=github)

Thank you for supporting open source software!

---

## 🏷️ License

GPL © [hello2himel](https://github.com/hello2himel)