FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv tzdata ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA (install before whisperx to prevent CPU-only torch)
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# Flash Attention 2 pre-built wheel for VibeVoice-ASR LLM decoder speedup.
# Applied via dict-form attn_implementation={"text_config": "flash_attention_2"}.
# Wheel must match: torch version + CUDA version + CXX11 ABI + Python version.
RUN pip3 install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

COPY omi_whisper_bridge.py .
COPY model_manager.py .
COPY term_corrector.py .
COPY llm_corrector.py .
COPY llm_pipeline.py .
COPY llm_pipeline_schema.py .
COPY llm_pipeline_prompt.py .
COPY batch_corrector.py .
COPY diarizer.py .
COPY audio_preprocessor.py .
COPY hotword_loader.py .
COPY speaker_filter.py .
COPY session_tagger.py .
COPY training_sanitizer.py .
COPY responses_sanitizer.py .
COPY suggestion_learner.py .
COPY segment_classifier.py .
COPY segment_screener.py .
COPY audio_scene_classifier.py .
COPY vibevoice_transcriber.py .
COPY correction_metrics.py .
COPY sanitize_provider.py .
COPY sanitize_schema.py .
COPY sanitize_prompt.py .
COPY test_whisper.py .

# モデルキャッシュを永続化するためのボリュームマウントポイント
ENV HF_HOME=/models

EXPOSE 8090
EXPOSE 8091

CMD ["python3", "omi_whisper_bridge.py"]
