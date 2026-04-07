FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv tzdata ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.4 (compatible with host driver 560.94 / CUDA 12.6)
RUN pip3 install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install all requirements (pyannote/whisperx will try to pull torch but pip
# will see the constraint is already satisfied by 2.6.0+cu124)
COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# Restore torch to cu124 if any dependency overwrote it
RUN pip3 install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


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
COPY rag_classifier.py .
COPY taxonomy_refiner.py .
COPY keyword_learner.py .
COPY training_sanitizer.py .
COPY responses_sanitizer.py .
COPY suggestion_learner.py .
COPY segment_classifier.py .
COPY segment_screener.py .
COPY audio_scene_classifier.py .
COPY vibevoice_transcriber.py .
COPY transcriber_qwen3asr.py .
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
