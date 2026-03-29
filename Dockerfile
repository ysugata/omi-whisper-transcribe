FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv tzdata ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA (install before whisperx to prevent CPU-only torch)
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# numpy first (needed by sox setup.py at install time)
RUN pip3 install --no-cache-dir numpy>=1.24.0

COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

COPY omi_whisper_bridge.py .
COPY model_manager.py .
COPY term_corrector.py .
COPY llm_corrector.py .
COPY diarizer.py .
COPY audio_preprocessor.py .
COPY hotword_loader.py .
COPY speaker_filter.py .
COPY session_tagger.py .
COPY asr_engine.py .
COPY whisper_engine.py .
COPY qwen_asr_engine.py .
COPY engine_manager.py .
COPY test_whisper.py .

# モデルキャッシュを永続化するためのボリュームマウントポイント
ENV HF_HOME=/models

EXPOSE 8090
EXPOSE 8091

CMD ["python3", "omi_whisper_bridge.py"]
