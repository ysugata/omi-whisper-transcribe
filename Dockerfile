FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

COPY omi_whisper_bridge.py .
COPY model_manager.py .
COPY test_whisper.py .

# モデルキャッシュを永続化するためのボリュームマウントポイント
ENV HF_HOME=/models

EXPOSE 8090
EXPOSE 8091

CMD ["python3", "omi_whisper_bridge.py"]
