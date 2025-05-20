# 必要なライブラリのインポート
from pyannote.audio import Pipeline
import whisper
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os
import torch
from scipy.io import wavfile

import glob

# dataディレクトリ内の全MP4/WAVファイルを取得
import itertools
mp4_files = glob.glob("audio/*.mp4")
wav_files = glob.glob("audio/*.wav")
media_files = list(itertools.chain(mp4_files, wav_files))

# 話者分離モデルの初期化
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))
except Exception as e:
    print("Warning: pyannote.audioのGPU初期化に失敗したためCPUで実行します。詳細:", e)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cpu"))

# Whisperモデルのロード
model = whisper.load_model("large-v3").to("cuda:1")

import os

# 出力ディレクトリ作成
os.makedirs("text", exist_ok=True)
import json

for media_path in media_files:
    base = os.path.splitext(os.path.basename(media_path))[0]
    ext = os.path.splitext(media_path)[1].lower()
    wav_path = f"audio/{base}.wav"
    output_json = f"text/{base}.json"
    results = []

    # MP4の場合はWAVに変換し、変換後にMP4を削除
    if ext == ".mp4":
        if not os.path.exists(wav_path):
            video = VideoFileClip(media_path)
            audio = video.audio
            audio.write_audiofile(wav_path, fps=16000, nbytes=2, codec='pcm_s16le')
            audio.close()
            video.close()
        # 変換後にmp4削除
        os.remove(media_path)

    # 音声ファイルを指定
    audio_file = wav_path if ext == ".mp4" else media_path  # mp4は変換後のwav、wavはそのまま

    # 話者分離の実行
    diarization = pipeline(audio_file)

    # WAVファイルをAudioSegmentで読み込む
    audio_segment = AudioSegment.from_file(audio_file, format="wav")

    # 音声ファイルを16kHz、モノラルに変換
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

    # ステレオWAV作成用の準備
    sample_rate = audio_segment.frame_rate
    total_samples = len(audio_segment.get_array_of_samples())
    duration_sec = len(audio_segment) / 1000.0
    stereo_array = np.zeros((int(sample_rate * duration_sec), 2), dtype=np.float32)  # [サンプル数, チャンネル]
    ab_segments = {"A": [], "B": []}

    # 話者分離の結果をループ処理
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        segment_audio = audio_segment[start_ms:end_ms]
        waveform = np.array(segment_audio.get_array_of_samples()).astype(np.float32)
        waveform = waveform / np.iinfo(segment_audio.array_type).max

        # 配置先チャンネル決定
        speaker_ab = "A" if speaker == "SPEAKER_00" else "B" if speaker == "SPEAKER_01" else speaker
        ab_segments[speaker_ab].append((segment.start, segment.end, waveform))

        # Whisperによる文字起こし（単語タイムスタンプ付き）
        result = model.transcribe(waveform, fp16=True, word_timestamps=True)

        for data in result["segments"]:
            start_time = segment.start + data["start"]
            end_time = segment.start + data["end"]
            if data["text"].strip():
                print(f"{start_time:.2f},{end_time:.2f},{speaker_ab},{data['text']}")
                results.append({
                    "speaker": speaker_ab,
                    "word": data["text"],
                    "start": start_time,
                    "end": end_time
                })

    # ステレオ配列にA/Bの音声を配置
    for speaker_ab, ch in zip(["A", "B"], [0, 1]):
        for seg_start, seg_end, waveform in ab_segments[speaker_ab]:
            start_idx = int(seg_start * sample_rate)
            end_idx = start_idx + len(waveform)
            # 配置範囲が配列を超えないように調整
            if end_idx > stereo_array.shape[0]:
                end_idx = stereo_array.shape[0]
                waveform = waveform[:end_idx - start_idx]
            stereo_array[start_idx:end_idx, ch] += waveform  # 重なりは加算

    # [-1.0, 1.0]をint16に変換
    stereo_array = np.clip(stereo_array, -1.0, 1.0)
    stereo_int16 = (stereo_array * 32767).astype(np.int16)

    # ステレオWAVとして保存
    stereo_wav_path = f"audio/{base}.wav"
    wavfile.write(stereo_wav_path, sample_rate, stereo_int16)

    # 結果をjsonファイルに保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)