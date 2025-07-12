#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper
import glob
import os

logger = logging.getLogger()


def process_single_audio(wav_path):
    # Check if text file exists
    txt_path = wav_path.replace('.wav', '.normalized.txt')
    if not os.path.exists(txt_path):
        logger.warning(f'{txt_path} does not exist, skipping {wav_path}')
        return None
    
    # Extract utterance ID
    utt = os.path.basename(wav_path).replace('.wav', '')
    
    # Process audio
    audio, sample_rate = torchaudio.load(wav_path, backend='soundfile')
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    
    # Convert audio to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    if audio.shape[1] / 16000 > 30:
        logging.warning(f'Audio longer than 30s, skipping tokenization for {wav_path}')
        speech_token = []
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        speech_token = ort_session.run(None, {
            ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
        })[0].flatten().tolist()
    
    # Save individual token file
    token_path = wav_path.replace('.wav', '_tokens.pt')
    torch.save(speech_token, token_path)
    
    return {
        'wav_path': wav_path,
        'utt': utt,
        'token_path': token_path,
        'num_tokens': len(speech_token)
    }


def main(args):
    # Find all wav files
    wav_files = list(glob.glob('{}/*/*/*wav'.format(args.src_dir)))
    print(f"Found {len(wav_files)} wav files")
    
    # Process all audio files
    all_tasks = [executor.submit(process_single_audio, wav_path) for wav_path in wav_files]
    
    # Collect results
    successful_files = []
    
    for future in tqdm(as_completed(all_tasks), total=len(all_tasks)):
        result = future.result()
        if result is None:
            continue
        successful_files.append(result)
    
    # Save a summary file for reference
    summary_path = os.path.join(args.src_dir, "token_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Processed {len(successful_files)} files successfully\n")
        total_tokens = sum(r['num_tokens'] for r in successful_files)
        f.write(f"Total tokens generated: {total_tokens}\n")
        for result in successful_files:
            f.write(f"{result['utt']} {result['wav_path']} {result['token_path']} {result['num_tokens']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="Source directory containing audio files")
    parser.add_argument("--onnx_path", type=str, help="Path to speech_tokenizer_v2.onnx model")
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)