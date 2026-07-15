# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# apt にも BuildKit cache mount を使い、archives/lists を再ダウンロードしない。
# cache mount を使う場合は rm -rf /var/lib/apt/lists/* 禁止 (cache と衝突)。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv tzdata ffmpeg wget \
    libopus0 libopus-dev

WORKDIR /app

# pip ダウンロードキャッシュを BuildKit cache mount に永続化。
# レイヤーキャッシュが外れても torch 等の wheel 再 HTTP DL を回避できる。
# --no-cache-dir は外す (cache mount がキャッシュ本体を保持する役割)。

# PyTorch with CUDA 12.4 (compatible with host driver 560.94 / CUDA 12.6)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install all requirements (pyannote/whisperx will try to pull torch but pip
# will see the constraint is already satisfied by 2.6.0+cu124)
COPY requirements-docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install -r requirements-docker.txt

# Restore torch to cu124 if any dependency overwrote it
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Qwen3-ASR: install with --no-deps to skip heavy/unnecessary deps (gradio, flask, sox, soynlp)
# then install only the runtime deps we actually need
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install --no-deps qwen-asr>=0.0.6 \
    && pip3 install qwen-omni-utils librosa nagisa

# Phonetic correction + text normalization (CPU only, no GPU)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install pykakasi neologdn


COPY omi_whisper_bridge.py .
COPY opus_decoder.py .
COPY sdcard_registry.py .
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
COPY phonetic_corrector.py .
COPY text_normalizer.py .
COPY correction_metrics.py .
COPY sanitize_provider.py .
COPY sanitize_schema.py .
COPY sanitize_prompt.py .
COPY succession_rag.py .
COPY wiki_index.py .
# SuccessionRAG imports exporter.SanitizedIndex from the viewer package;
# copy the whole subdirectory so the sys.path.insert at module top works.
COPY omi-recordings-viewer/exporter.py ./omi-recordings-viewer/exporter.py
COPY test_whisper.py .

# モデルキャッシュを永続化するためのボリュームマウントポイント
ENV HF_HOME=/models

EXPOSE 8090
EXPOSE 8091

CMD ["python3", "omi_whisper_bridge.py"]
