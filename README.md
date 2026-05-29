<div align="center">

# 🎬 Video RAG Pipeline
### *Turn any video course into a conversational AI tutor*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Audio%20Processing-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)](https://ollama.com)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=for-the-badge)]()

<br/>

> **Ask questions about your entire video course — and get answers with exact video numbers and timestamps.**
> Built on top of *Code With Harry's* Data Science course as a real-world RAG use case.

</div>

---

## 🗺️ Pipeline Architecture

![RAG Pipeline Architecture](pipeline.svg)

*End-to-end pipeline: from raw `.mp4` files to LLM-powered Q&A with timestamped citations*

Each chunk produced by the pipeline carries rich metadata:

```json
{
  "video_number": "01",
  "video_title": "what is data science",
  "start": 0.0,
  "end": 5.42,
  "text": " Data science is the study of data to extract meaningful insights..."
}
```

---

## ✅ Progress Tracker

| Step | Description | Tool / Model | Status |
|------|-------------|--------------|--------|
| **Step 1** | Videos → Audio (extraction) | FFmpeg | ✅ Done |
| **Step 2** | Audio → Transcription + Chunking | OpenAI Whisper `medium` | ✅ Done |
| **Step 3** | Text → Vectors (embeddings) | `bge-m3` via Ollama | ✅ Done |
| **Step 4** | Query → Semantic Search | cosine similarity (sklearn) | ✅ Done |
| **Step 5** | Prompt construction + LLM inference | Ollama (`qwen2.5:7b` / `deepseek-r1`) | ✅ Done |
| **Step 6** | Chat UI | Streamlit / Gradio | ⏳ Pending |

---

## 📁 Project Structure

```
video-rag-pipeline/
│
├── videos/                    # Raw .mp4 lecture files
│   └── 01_what_is_data_science.mp4
│
├── audios/                    # Extracted .mp3 files (via FFmpeg)
│   └── 01_what_is_data_science.mp3
│
├── jsons/                     # Whisper transcript chunks with metadata
│   └── 01_what_is_data_science.mp3.json
│
├── pipeline/
│   ├── video_to_audio.py      # Step 1 — FFmpeg: .mp4 → .mp3
│   ├── audio_to_chunks.py     # Step 2 — Whisper: .mp3 → timestamped JSON
│   ├── read_chunks.py         # Step 3 — Embed all chunks → save .joblib
│   └── process_incoming.py    # Step 4+5 — Query → retrieve → LLM answer
│
├── chunks_embeddings.joblib   # Saved DataFrame with all embeddings
├── prompt.txt                 # Last generated prompt (for debugging)
├── response.txt               # Last LLM response (for debugging)
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/video-rag-pipeline.git
cd video-rag-pipeline
```

### 2. Install Python dependencies

```bash
pip install openai-whisper scikit-learn pandas numpy joblib requests
```

### 3. Install FFmpeg

FFmpeg is required to extract audio from video files.

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add the bin/ folder to your system PATH
```

### 4. Install Ollama + pull models

Ollama runs LLMs locally on your machine — no API keys needed.

```bash
# Install Ollama from https://ollama.com

# Pull the embedding model
ollama pull bge-m3

# Pull an LLM for inference (choose based on your RAM)
ollama pull qwen2.5:7b        # Recommended — ~4.7GB, great quality
ollama pull deepseek-r1:7b    # Alternative — ~4.7GB, strong reasoning
ollama pull llama3.2:3b       # Lightweight — ~2GB, if RAM is tight
```

> **Note on Whisper models:** The pipeline uses `medium` by default which gives a good balance of speed and accuracy. If you want better transcription quality (especially for accented speech), switch to `large` or `large-v3`. If you want faster processing, use `small` or `base`. Change `whisper.load_model("medium")` in `audio_to_chunks.py` accordingly.
>
> | Model | Size | Speed | Accuracy |
> |-------|------|-------|----------|
> | `base` | 74MB | ⚡⚡⚡ | ⭐⭐ |
> | `small` | 244MB | ⚡⚡ | ⭐⭐⭐ |
> | `medium` | 769MB | ⚡ | ⭐⭐⭐⭐ ← default |
> | `large-v3` | 1.5GB | 🐢 | ⭐⭐⭐⭐⭐ |

---

## 🚀 Usage

### Step 1 — Extract Audio from Videos

Extracts audio from all `.mp4` files in `videos/` and saves as `.mp3` in `audios/`.

```bash
python pipeline/video_to_audio.py
```

```python
# pipeline/video_to_audio.py
import os, subprocess

os.makedirs('audios', exist_ok=True)

for file in os.listdir('videos'):
    video_number = file.split('_')[0]
    video_name = '_'.join(file.split('_')[1:]).replace('.mp4', '')
    subprocess.run(['ffmpeg', '-i', f"videos/{file}", f"audios/{video_number}_{video_name}.mp3"])
```

---

### Step 2 — Transcribe & Chunk Audio

Runs Whisper on every audio file, translates to English, and saves timestamped chunks as JSON.

```bash
python pipeline/audio_to_chunks.py
```

Output format — one JSON per video:
```json
{
  "chunks": [
    {
      "video_number": "01",
      "video_title": "what is data science",
      "start": 0.0,
      "end": 5.42,
      "text": " Data science is the study of data to extract meaningful insights..."
    }
  ],
  "text": "Full transcript text..."
}
```

---

### Step 3 — Generate Embeddings

Embeds every chunk using `bge-m3` via Ollama and saves everything to a `.joblib` file.

```bash
python pipeline/read_chunks.py
```

This only needs to be run once. The output `chunks_embeddings.joblib` is a pandas DataFrame with all chunk metadata + embedding vectors.

---

### Step 4+5 — Ask Questions

```bash
python pipeline/process_incoming.py
```

```
Ask Anything: What skills do I need to become a data scientist?
```

The script will:
1. Embed your query using `bge-m3`
2. Find the top 30 most similar chunks via cosine similarity
3. Build a structured prompt with the relevant chunks
4. Send it to your local Ollama LLM
5. Print and save the response with video numbers and timestamps

---

## 🧠 Design Decisions

| Choice | Reason |
|--------|--------|
| **Whisper `medium`** | Best balance of speed and accuracy for Hindi-accented English |
| `task="translate"` | Normalizes any Hindi segments into English for consistent embeddings |
| **`bge-m3` embeddings** | State-of-the-art multilingual embedding model, runs locally |
| **Segment-level chunking** | Preserves natural sentence boundaries and gives exact timestamps |
| **`qwen2.5:7b` for inference** | Strong instruction following in a small package — runs on 6GB RAM |
| **joblib for storage** | Simple, fast, and preserves numpy arrays perfectly — no vector DB needed at this scale |
| **Top-30 retrieval** | Gives the LLM enough context without blowing the context window |

---

## 💡 Choosing the Right Ollama LLM

| Model | RAM Needed | Best For |
|-------|-----------|----------|
| `llama3.2:3b` | ~3GB | Very low RAM, decent answers |
| `qwen2.5:7b` | ~5GB | **Best for most people** — fast + accurate |
| `deepseek-r1:7b` | ~5GB | Better reasoning, slightly slower |
| `mistral-small3.1:24b` | ~16GB | High quality if you have the RAM |
| `deepseek-r1:32b` | ~20GB | Best quality, needs 20GB+ RAM |

---

## 🔮 Roadmap

- [x] Step 1 — FFmpeg video → audio extraction
- [x] Step 2 — Whisper transcription + JSON chunking
- [x] Step 3 — bge-m3 embeddings + joblib storage
- [x] Step 4 — Cosine similarity semantic search
- [x] Step 5 — RAG prompt + Ollama LLM inference
- [ ] Step 6 — Streamlit / Gradio chat UI
- [ ] Add support for YouTube video URLs (auto-download + process)
- [ ] Swap joblib for ChromaDB / FAISS for larger course libraries
- [ ] Multi-course support with course-level filtering

---

## 🙏 Acknowledgements

- **[Code With Harry](https://www.codewithharry.com/)** — for the original Data Science course used as the dataset
- **[OpenAI Whisper](https://github.com/openai/whisper)** — incredible open-source transcription model
- **[FFmpeg](https://ffmpeg.org)** — reliable, fast audio extraction
- **[Ollama](https://ollama.com)** — making local LLMs dead simple to run
- **[bge-m3](https://huggingface.co/BAAI/bge-m3)** — state-of-the-art multilingual embeddings

---

<div align="center">

Made with 🔥 while grinding through a Data Science course

*Turning passive watching into active, searchable knowledge*

</div>
