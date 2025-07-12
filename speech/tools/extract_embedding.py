#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
import os
import glob
import logging

logger = logging.getLogger()


def process_single_audio(wav_path):
    # Extract utterance ID and speaker ID from filename
    utt = os.path.basename(wav_path).replace('.wav', '')
    spk = utt.split('_')[0]
    
    # Check if text file exists
    txt_path = wav_path.replace('.wav', '.normalized.txt')
    if not os.path.exists(txt_path):
        logger.warning(f'{txt_path} does not exist, skipping {wav_path}')
        return None
    
    # Process audio
    audio, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    # Generate embedding
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
    
    # Save individual embedding file
    embedding_path = wav_path.replace('.wav', '_embedding.pt')
    torch.save(embedding, embedding_path)
    
    return {
        'wav_path': wav_path,
        'utt': utt,
        'spk': spk,
        'embedding': embedding,
        'embedding_path': embedding_path
    }


def main(args):
    # Find all wav files
    wav_files = list(glob.glob('{}/*/*/*wav'.format(args.src_dir)))
    print(f"Found {len(wav_files)} wav files")
    
    # Process all audio files
    all_tasks = [executor.submit(process_single_audio, wav_path) for wav_path in wav_files]
    
    # Collect results
    spk2embeddings = {}
    successful_files = []
    
    for future in tqdm(as_completed(all_tasks), total=len(all_tasks)):
        result = future.result()
        if result is None:
            continue
            
        successful_files.append(result)
        
        # Collect embeddings by speaker
        spk = result['spk']
        if spk not in spk2embeddings:
            spk2embeddings[spk] = []
        spk2embeddings[spk].append(result['embedding'])
    
    # Calculate and save speaker embeddings
    spk_embed_dir = os.path.join(args.src_dir, "spk_embeddings")
    os.makedirs(spk_embed_dir, exist_ok=True)
    
    for spk, embeddings in spk2embeddings.items():
        spk_embedding = torch.stack([torch.tensor(e) for e in embeddings]).mean(dim=0)
        spk_embedding_path = os.path.join(spk_embed_dir, f"{spk}_embedding.pt")
        torch.save(spk_embedding, spk_embedding_path)
        print(f"Saved speaker embedding for {spk} with {len(embeddings)} utterances")
    
    # Save a summary file for reference
    summary_path = os.path.join(args.src_dir, "embedding_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Processed {len(successful_files)} files successfully\n")
        f.write(f"Found {len(spk2embeddings)} speakers\n")
        for result in successful_files:
            f.write(f"{result['utt']} {result['wav_path']} {result['embedding_path']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="Source directory containing audio files")
    parser.add_argument("--onnx_path", type=str, help="Path to campplus.onnx model")
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)