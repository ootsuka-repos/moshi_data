# 必要なライブラリのインポート
from pyannote.audio import Pipeline
# import whisper # whisperライブラリは使用しないためコメントアウト
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os
import torch
from scipy.io import wavfile # wavfileのインポートを再度有効化
from transformers import pipeline as hf_pipeline # pipelineという名前が衝突するためエイリアスを設定
# from datasets import load_dataset # 今回はローカルファイルを使用するためコメントアウト

import glob

# dataディレクトリ内の全MP4/WAVファイルを取得
import itertools
mp4_files = glob.glob("inputs/*.mp4") # 入力ディレクトリを inputs/ に変更
wav_files = glob.glob("inputs/*.wav") # 入力ディレクトリを inputs/ に変更
media_files = list(itertools.chain(mp4_files, wav_files))

# config for kotoba-whisper
model_id = "kotoba-tech/kotoba-whisper-v2.1"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu" # pyannoteとデバイスを合わせるか検討
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "ja", "task": "transcribe"}


try:
    pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    # Whisperモデルが "cuda:0" を使う場合、pyannoteも同じGPUか、別のGPU (例: "cuda:1") を指定
    # ここではWhisperと同じdeviceを使うように設定
    pyannote_pipeline.to(torch.device(device))
except Exception as e:
    print(f"Warning: pyannote.audioのGPU初期化に失敗したためCPUで実行します。詳細: {e}")
    pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pyannote_pipeline.to(torch.device("cpu"))


# Whisperモデルのロード (transformers pipelineを使用)
# model = whisper.load_model("large-v3").to("cuda:1") # 従来のwhisperロードはコメントアウト
transcribe_pipe = hf_pipeline(
    model=model_id,
    torch_dtype=torch_dtype,
    device=device, # pyannoteと同じデバイスを使用
    model_kwargs=model_kwargs,
    batch_size=16, # 必要に応じて調整
    trust_remote_code=True,
    # punctuator=True # kotoba-whisperのpipelineではpunctuator引数がない場合がある
)


import os

# 出力ディレクトリ作成
os.makedirs("text", exist_ok=True)
import json

for media_path in media_files:
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    file_ext = os.path.splitext(media_path)[1].lower()
    
    output_wav_path = f"audio/{base_name}.wav" # 出力WAVファイルパス
    output_json_path = f"text/{base_name}.json" # 出力JSONファイルパス
    
    results = []
    ab_segments = {"A": [], "B": []} # 話者ごとの音声セグメントを保存

    # audio と text ディレクトリがなければ作成
    os.makedirs("audio", exist_ok=True)
    os.makedirs("text", exist_ok=True)

    processed_audio_path = media_path # 初期値は入力パス
    temp_conversion_wav_path = None # MP4から変換した一時WAVのパス

    if file_ext == ".mp4":
        temp_conversion_wav_path = f"audio/temp_conversion_{base_name}.wav"
        print(f"Converting MP4 '{media_path}' to WAV '{temp_conversion_wav_path}'")
        try:
            video = VideoFileClip(media_path)
            audio = video.audio
            # 16kHz, モノラル, s16leで保存
            audio.write_audiofile(temp_conversion_wav_path, fps=16000, nbytes=2, codec='pcm_s16le')
            audio.close()
            video.close()
            processed_audio_path = temp_conversion_wav_path
        except Exception as e:
            print(f"Error converting MP4 {media_path}: {e}")
            if temp_conversion_wav_path and os.path.exists(temp_conversion_wav_path):
                os.remove(temp_conversion_wav_path)
            continue # このファイルの処理をスキップ
    elif file_ext != ".wav":
        print(f"Skipping unsupported file type: {media_path}")
        continue

    print(f"Processing file: {processed_audio_path}")
    # WAVファイルをAudioSegmentで読み込む
    try:
        audio_segment_full = AudioSegment.from_file(processed_audio_path, format="wav")
    except Exception as e:
        print(f"Error loading audio file {processed_audio_path}: {e}")
        if temp_conversion_wav_path and os.path.exists(temp_conversion_wav_path):
            os.remove(temp_conversion_wav_path)
        continue
    
    # 音声ファイルを16kHz、モノラルに変換 (MP4からの変換時に実施済みだが念のため)
    audio_segment_full = audio_segment_full.set_frame_rate(16000).set_channels(1)

    if len(audio_segment_full) == 0:
        print(f"Skipping empty audio file: {processed_audio_path}")
        if temp_conversion_wav_path and os.path.exists(temp_conversion_wav_path):
            os.remove(temp_conversion_wav_path)
        continue

    # 話者分離の実行 (ファイル全体に対して)
    # pyannoteに渡すために、AudioSegmentを一時ファイルとして保存する必要がある場合がある。
    # processed_audio_path が一時ファイルでない場合（元のWAVファイルの場合）、
    # それをpyannoteに渡せるか、あるいは一時的なモノラルWAVを作成するか検討。
    # ここでは、MP4からの変換で既に一時WAVになっているか、元のWAVパスをそのまま使用。
    # ただし、pyannoteがAudioSegmentオブジェクトを直接受け付けないため、
    # 常にファイルパス（processed_audio_path）を使用する。
    # もしprocessed_audio_pathが元のWAVで、pyannoteがそれをうまく扱えない場合、
    # ここで audio_segment_full を一時ファイルにエクスポートする必要が出てくる。
    # 現状は processed_audio_path を信じる。

    diarization_input_path = processed_audio_path
    # もし pyannote が AudioSegment から直接処理できず、かつ元のWAVファイルがステレオ等の場合、
    # ここでモノラル16kHzの一時ファイルを作成する。
    # audio_segment_full は既にモノラル16kHzなので、これを一時ファイルに書き出す。
    temp_diarization_input_path = f"audio/temp_diarization_input_{base_name}.wav"
    audio_segment_full.export(temp_diarization_input_path, format="wav")


    try:
        print(f"Running diarization on: {temp_diarization_input_path}")
        diarization = pyannote_pipeline(temp_diarization_input_path)
    except Exception as e:
        print(f"Error during diarization for file {temp_diarization_input_path}: {e}")
        if temp_conversion_wav_path and os.path.exists(temp_conversion_wav_path):
            os.remove(temp_conversion_wav_path)
        if os.path.exists(temp_diarization_input_path):
            os.remove(temp_diarization_input_path)
        continue
    finally:
        if os.path.exists(temp_diarization_input_path):
            os.remove(temp_diarization_input_path)


    # ステレオWAV作成用の準備 (ファイル全体に対して)
    sample_rate = audio_segment_full.frame_rate
    duration_sec = len(audio_segment_full) / 1000.0
    stereo_array = np.zeros((int(sample_rate * duration_sec) + 100, 2), dtype=np.float32) # 少しバッファを持たせる

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in ["SPEAKER_00", "SPEAKER_01"]:
            continue

        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)

        start_ms = max(0, start_ms)
        end_ms = min(len(audio_segment_full), end_ms)
        if start_ms >= end_ms:
            continue
        
        segment_audio_pydub = audio_segment_full[start_ms:end_ms]

        if len(segment_audio_pydub) == 0:
            continue

        samples = np.array(segment_audio_pydub.get_array_of_samples()).astype(np.float32)
        if samples.size == 0: continue
        samples /= np.iinfo(segment_audio_pydub.array_type).max

        speaker_ab = "A" if speaker == "SPEAKER_00" else "B"
        # ab_segments はファイル単位なので、初期化されたものに追加
        if speaker_ab not in ab_segments: # 通常は発生しないはず
            ab_segments[speaker_ab] = []
        ab_segments[speaker_ab].append((segment.start, segment.end, samples))


        transcription_result = transcribe_pipe(
            {"raw": samples, "sampling_rate": segment_audio_pydub.frame_rate},
            chunk_length_s=15,
            return_timestamps=True,
            generate_kwargs=generate_kwargs.copy()
        )

        if transcription_result and "chunks" in transcription_result:
            for chunk_data_item in transcription_result["chunks"]:
                word_text = chunk_data_item["text"]
                if chunk_data_item["timestamp"]:
                    # タイムスタンプはセグメントの先頭からなので、ファイル全体の時間に変換
                    word_start_time = segment.start + chunk_data_item["timestamp"][0]
                    word_end_time = segment.start + chunk_data_item["timestamp"][1]
                    if word_text.strip():
                        results.append({
                            "speaker": speaker_ab,
                            "word": word_text,
                            "start": round(word_start_time, 3),
                            "end": round(word_end_time, 3)
                        })
        elif transcription_result and "text" in transcription_result:
            full_text = transcription_result["text"]
            if full_text.strip():
                results.append({
                    "speaker": speaker_ab,
                    "word": full_text,
                    "start": round(segment.start, 3), # セグメント全体の開始終了
                    "end": round(segment.end, 3)
                })
    
    results.sort(key=lambda x: x["start"])

    for speaker_label, ch_idx in zip(["A", "B"], [0, 1]):
        for seg_start_sec, seg_end_sec, waveform_data in ab_segments.get(speaker_label, []):
            start_sample_idx = int(seg_start_sec * sample_rate)
            end_sample_idx = start_sample_idx + len(waveform_data)

            if start_sample_idx >= stereo_array.shape[0]: continue
            
            # 波形データの長さに合わせてスライス範囲を調整
            current_waveform_len = len(waveform_data)
            effective_end_sample_idx = start_sample_idx + current_waveform_len

            if effective_end_sample_idx > stereo_array.shape[0]:
                # print(f"Warning: Waveform for speaker {speaker_label} (start {seg_start_sec:.2f}s, len {current_waveform_len}) exceeds stereo_array bounds ({stereo_array.shape[0]}). Truncating.")
                waveform_data = waveform_data[:stereo_array.shape[0] - start_sample_idx]
                effective_end_sample_idx = stereo_array.shape[0]
            
            if len(waveform_data) > 0 :
                stereo_array[start_sample_idx:effective_end_sample_idx, ch_idx] += waveform_data
    
    # stereo_array の末尾の余分なゼロをトリム (バッファ分)
    # 実際にデータが書き込まれた最大のインデックスを見つける
    last_written_idx = 0
    for speaker_label in ["A", "B"]:
        for seg_start_sec, _, waveform_data in ab_segments.get(speaker_label, []):
            idx = int(seg_start_sec * sample_rate) + len(waveform_data)
            if idx > last_written_idx:
                last_written_idx = idx
    
    # 実際の音声長に合わせて stereo_array をトリム
    # last_written_idx が stereo_array の長さを超えることはないはずだが、念のため min を取る
    actual_len_samples = min(last_written_idx, stereo_array.shape[0])
    stereo_array_trimmed = stereo_array[:actual_len_samples, :]


    stereo_array_trimmed = np.clip(stereo_array_trimmed, -1.0, 1.0)
    stereo_int16 = (stereo_array_trimmed * 32767).astype(np.int16)
    
    if stereo_int16.size == 0:
        print(f"Warning: Resulting stereo audio for {base_name} is empty. Skipping WAV write.")
    else:
        try:
            wavfile.write(output_wav_path, sample_rate, stereo_int16)
            print(f"Successfully wrote stereo WAV: {output_wav_path}")
        except Exception as e:
            print(f"Error writing WAV file {output_wav_path}: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Successfully wrote JSON: {output_json_path}")

    # MP4から変換した一時ファイルを削除
    if temp_conversion_wav_path and os.path.exists(temp_conversion_wav_path):
        os.remove(temp_conversion_wav_path)
        print(f"Deleted temporary conversion WAV: {temp_conversion_wav_path}")
    
    # 元のMP4ファイルを削除する場合のロジック（必要ならコメント解除）
    # if file_ext == ".mp4" and os.path.exists(media_path):
    #     print(f"Deleting original MP4 file: {media_path}")
    #     os.remove(media_path)