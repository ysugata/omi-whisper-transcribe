"""
faster-whisper 動作テストスクリプト

ブリッジアプリを起動する前に、このスクリプトで
GPU でモデルが正しく動くか確認してください。

使い方:
  python test_whisper.py
  python test_whisper.py --model medium --audio test.wav
"""

import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="faster-whisper 動作テスト")
    parser.add_argument("--model", default="large-v3", help="モデル名")
    parser.add_argument("--compute", default="int8_float16", help="計算精度")
    parser.add_argument("--audio", default=None, help="テスト用音声ファイル (省略時はダミー音声)")
    parser.add_argument("--language", default="ja", help="言語コード")
    args = parser.parse_args()

    print(f"=== faster-whisper 動作テスト ===")
    print(f"Model: {args.model}")
    print(f"Compute: {args.compute}")
    print()

    # Step 1: CUDA チェック
    print("[1/3] CUDA チェック...")
    try:
        import ctranslate2
        print(f"  ctranslate2 version: {ctranslate2.__version__}")
        print(f"  CUDA available: {ctranslate2.get_cuda_device_count() > 0}")
        print(f"  CUDA devices: {ctranslate2.get_cuda_device_count()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  → CUDA ライブラリが見つかりません。LD_LIBRARY_PATH を確認してください。")
        sys.exit(1)

    # Step 2: モデル読み込み
    print(f"\n[2/3] モデル読み込み ({args.model})...")
    t0 = time.time()
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cuda", compute_type=args.compute)
        load_time = time.time() - t0
        print(f"  OK — {load_time:.1f}秒で読み込み完了")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Step 3: 文字起こしテスト
    print(f"\n[3/3] 文字起こしテスト...")
    if args.audio:
        print(f"  ファイル: {args.audio}")
        t0 = time.time()
        segments, info = model.transcribe(
            args.audio,
            beam_size=5,
            language=args.language,
            vad_filter=True,
        )
        results = list(segments)
        elapsed = time.time() - t0

        print(f"  検出言語: {info.language} (確率: {info.language_probability:.2f})")
        print(f"  セグメント数: {len(results)}")
        print(f"  処理時間: {elapsed:.2f}秒")
        print()
        for seg in results:
            print(f"  [{seg.start:.2f}s → {seg.end:.2f}s] {seg.text}")
    else:
        # ダミー音声 (1秒の無音) でエラーなく動くか確認
        import numpy as np
        dummy = np.zeros(16000, dtype=np.float32)
        t0 = time.time()
        segments, info = model.transcribe(dummy, beam_size=1, language=args.language)
        _ = list(segments)  # generator を消費
        elapsed = time.time() - t0
        print(f"  ダミー音声 (1秒無音) → OK ({elapsed:.2f}秒)")
        print(f"  ※ 実際の音声ファイルでテストするには: python test_whisper.py --audio your_file.wav")

    # VRAM 使用量
    print()
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(", ")
            print(f"GPU VRAM: {used}MB / {total}MB")
    except Exception:
        pass

    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    main()
