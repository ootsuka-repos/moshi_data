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
mp4_files = glob.glob("audio/*.mp4")
wav_files = glob.glob("audio/*.wav")
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
            # 16kHz, モノラル, s16leで保存 (Whisperの期待する形式)
            audio.write_audiofile(wav_path, fps=16000, nbytes=2, codec='pcm_s16le')
            audio.close()
            video.close()
        # 変換後にmp4削除
        if os.path.exists(media_path): # 念のため存在確認
            os.remove(media_path)

    # 音声ファイルを指定
    audio_file = wav_path if ext == ".mp4" or not media_path.endswith(".wav") else media_path

    # # 話者分離の実行 (ファイル全体ではなくチャンクごとに行う)
    # diarization = pyannote_pipeline(audio_file) # この行はチャンク処理ループ内に移動

    # WAVファイルをAudioSegmentで読み込む
    full_audio_segment = AudioSegment.from_file(audio_file, format="wav")

    # 音声ファイルを16kHz、モノラルに変換 (MP4からの変換時に実施済みだが念のため)
    full_audio_segment = full_audio_segment.set_frame_rate(16000).set_channels(1)

    # ステレオWAV作成用の準備 (ファイル全体に対して)
    sample_rate = full_audio_segment.frame_rate
    # total_samples = len(full_audio_segment.get_array_of_samples()) # 不要になる
    duration_sec_total = len(full_audio_segment) / 1000.0
    stereo_array = np.zeros((int(sample_rate * duration_sec_total), 2), dtype=np.float32)
    ab_segments = {"A": [], "B": []} # 話者ごとの音声セグメントを保存するために再度有効化

    chunk_duration_ms = 3 * 60 * 1000  # 3 minutes in milliseconds
    num_chunks = int(np.ceil(len(full_audio_segment) / chunk_duration_ms))

    for i in range(num_chunks):
        chunk_start_ms = i * chunk_duration_ms
        chunk_end_ms = min((i + 1) * chunk_duration_ms, len(full_audio_segment))
        current_chunk_audio_pydub = full_audio_segment[chunk_start_ms:chunk_end_ms]
        
        print(f"Processing chunk {i+1}/{num_chunks} ({chunk_start_ms/1000:.2f}s to {chunk_end_ms/1000:.2f}s of original audio)")

        # pyannoteに渡すために一時ファイルとしてチャンクを保存
        temp_chunk_path = f"audio/temp_chunk_{base}_{i}.wav"
        current_chunk_audio_pydub.export(temp_chunk_path, format="wav")

        try:
            # 話者分離の実行 (現在のチャンクに対して)
            diarization_chunk = pyannote_pipeline(temp_chunk_path)
        finally:
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path) # 一時ファイルを削除

        # 話者分離の結果をループ処理 (現在のチャンクの分離結果に対して)
        for segment, _, speaker in diarization_chunk.itertracks(yield_label=True):
            # segment.start, segment.end はチャンクの先頭からの相対時間(秒)
            segment_start_in_chunk_ms = int(segment.start * 1000)
            segment_end_in_chunk_ms = int(segment.end * 1000)

            # チャンク内の音声セグメントを抽出
            segment_audio_for_transcribe_pydub = current_chunk_audio_pydub[segment_start_in_chunk_ms:segment_end_in_chunk_ms]
            
            # pydub AudioSegment を numpy 配列に変換 (float32, -1.0 to 1.0)
            samples = np.array(segment_audio_for_transcribe_pydub.get_array_of_samples()).astype(np.float32)
            if samples.size == 0: # 空のセグメントはスキップ
                continue
            samples /= np.iinfo(segment_audio_for_transcribe_pydub.array_type).max # Normalize to -1.0 to 1.0

            # 配置先チャンネル決定
            speaker_ab = "A" if speaker == "SPEAKER_00" else "B" if speaker == "SPEAKER_01" else speaker
            
            # ステレオWAV再構築用に、絶対時間と対応する波形データを保存
            # segment.start/end はチャンクの開始からの秒数
            abs_segment_start_sec = (chunk_start_ms / 1000.0) + segment.start
            abs_segment_end_sec = (chunk_start_ms / 1000.0) + segment.end
            ab_segments[speaker_ab].append((abs_segment_start_sec, abs_segment_end_sec, samples)) # samples を保存

            # Whisper (transformers pipeline) による文字起こし
            current_generate_kwargs = generate_kwargs.copy()
            transcription_result = transcribe_pipe(
                {"raw": samples, "sampling_rate": segment_audio_for_transcribe_pydub.frame_rate},
                chunk_length_s=15,
                return_timestamps=True,
                generate_kwargs=current_generate_kwargs
            )

            # transcription_result の形式を確認し、適切に処理する
            if transcription_result and "chunks" in transcription_result:
                for chunk_data_item in transcription_result["chunks"]: # 変数名を変更
                    word_text = chunk_data_item["text"]
                    if chunk_data_item["timestamp"]:
                        # chunk_data_item["timestamp"] は文字起こしされたセグメントの先頭からの相対時間
                        word_start_rel_to_segment_sec = chunk_data_item["timestamp"][0]
                        word_end_rel_to_segment_sec = chunk_data_item["timestamp"][1]
                        
                        # 絶対時間を計算
                        abs_word_start_time = abs_segment_start_sec + word_start_rel_to_segment_sec
                        abs_word_end_time = abs_segment_start_sec + word_end_rel_to_segment_sec
                        
                        if word_text.strip():
                            print(f"{abs_word_start_time:.2f},{abs_word_end_time:.2f},{speaker_ab},{word_text}")
                            results.append({
                                "speaker": speaker_ab,
                                "word": word_text,
                                "start": abs_word_start_time,
                                "end": abs_word_end_time
                            })
            elif transcription_result and "text" in transcription_result:
                full_text = transcription_result["text"]
                if full_text.strip():
                    # この場合のタイムスタンプはセグメント全体のものを利用
                    print(f"{abs_segment_start_sec:.2f},{abs_segment_end_sec:.2f},{speaker_ab},{full_text} (no word timestamps)")
                    results.append({
                        "speaker": speaker_ab,
                        "word": full_text,
                        "start": abs_segment_start_sec,
                        "end": abs_segment_end_sec
                    })
    
    # results を 'start' キーでソート (チャンク処理により順序が乱れる可能性があるため)
    results.sort(key=lambda x: x["start"])

    # ステレオ配列にA/Bの音声を配置 (全体の音声に対して)
    for speaker_label, ch_idx in zip(["A", "B"], [0, 1]):
        for seg_start_sec, seg_end_sec, waveform_data in ab_segments.get(speaker_label, []):
            start_sample_idx = int(seg_start_sec * sample_rate)
            # waveform_data はセグメントの音声データなので、その長さを使う
            end_sample_idx = start_sample_idx + len(waveform_data)

            # 配列の範囲内に収まるように調整
            if start_sample_idx >= stereo_array.shape[0]:
                continue
            if end_sample_idx > stereo_array.shape[0]:
                waveform_data = waveform_data[:stereo_array.shape[0] - start_sample_idx]
                end_sample_idx = stereo_array.shape[0]
            
            if len(waveform_data) > 0:
                # ターゲットスライスの長さを再確認
                current_target_slice_len = stereo_array[start_sample_idx:end_sample_idx, ch_idx].shape[0]
                if len(waveform_data) > current_target_slice_len:
                     waveform_data = waveform_data[:current_target_slice_len]
                
                # waveform_data の長さに合わせて end_sample_idx を調整
                # これにより、加算時の形状不一致を防ぐ
                effective_end_sample_idx = start_sample_idx + len(waveform_data)

                stereo_array[start_sample_idx:effective_end_sample_idx, ch_idx] += waveform_data


    # [-1.0, 1.0]をint16に変換
    stereo_array = np.clip(stereo_array, -1.0, 1.0)
    stereo_int16 = (stereo_array * 32767).astype(np.int16)

    # ステレオWAVとして保存
    stereo_wav_path = f"audio/{base}.wav" # 元のファイル名で上書き
    wavfile.write(stereo_wav_path, sample_rate, stereo_int16)

    # 結果をjsonファイルに保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)