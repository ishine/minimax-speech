#!/usr/bin/env python3
"""Validate that all required files exist for training"""

import argparse
import glob
import os
from tqdm import tqdm

def validate_data(src_dir):
    """Check that all required files exist
    
    Args:
        src_dir: Directory containing audio files
    """
    # Find all wav files
    wav_files = glob.glob(os.path.join(src_dir, '*/*/*wav'))
    if not wav_files:
        wav_files = glob.glob(os.path.join(src_dir, '**/*.wav'), recursive=True)
    
    print(f"Found {len(wav_files)} WAV files")
    
    missing_txt = []
    missing_embedding = []
    missing_token = []
    missing_spk_embedding = []
    speakers = set()
    
    for wav_path in tqdm(wav_files, desc="Validating files"):
        # Check text file
        txt_path = wav_path.replace('.wav', '.normalized.txt')
        if not os.path.exists(txt_path):
            missing_txt.append(wav_path)
        
        # Check embedding file
        embedding_path = wav_path.replace('.wav', '_embedding.pt')
        if not os.path.exists(embedding_path):
            missing_embedding.append(wav_path)
        
        # Check token file
        token_path = wav_path.replace('.wav', '_tokens.pt')
        if not os.path.exists(token_path):
            missing_token.append(wav_path)
        
        # Extract speaker
        utt = os.path.basename(wav_path).replace('.wav', '')
        spk = utt.split('_')[0]
        speakers.add(spk)
    
    # Check speaker embeddings
    spk_embed_dir = os.path.join(src_dir, 'spk_embeddings')
    if os.path.exists(spk_embed_dir):
        for spk in speakers:
            spk_embedding_path = os.path.join(spk_embed_dir, f'{spk}_embedding.pt')
            if not os.path.exists(spk_embedding_path):
                missing_spk_embedding.append(spk)
    else:
        print(f"Speaker embedding directory not found: {spk_embed_dir}")
        missing_spk_embedding = list(speakers)
    
    # Report results
    print("\n=== Validation Results ===")
    print(f"Total WAV files: {len(wav_files)}")
    print(f"Total speakers: {len(speakers)}")
    print(f"Missing text files: {len(missing_txt)}")
    print(f"Missing embedding files: {len(missing_embedding)}")
    print(f"Missing token files: {len(missing_token)}")
    print(f"Missing speaker embeddings: {len(missing_spk_embedding)}")
    
    if missing_txt:
        print(f"\nFirst 5 missing text files:")
        for f in missing_txt[:5]:
            print(f"  {f}")
    
    if missing_embedding:
        print(f"\nFirst 5 missing embedding files:")
        for f in missing_embedding[:5]:
            print(f"  {f}")
    
    if missing_token:
        print(f"\nFirst 5 missing token files:")
        for f in missing_token[:5]:
            print(f"  {f}")
    
    if missing_spk_embedding:
        print(f"\nFirst 5 missing speaker embeddings:")
        for spk in list(missing_spk_embedding)[:5]:
            print(f"  {spk}")
    
    # Return success if no missing files
    return len(missing_txt) == 0 and len(missing_embedding) == 0 and len(missing_token) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True,
                       help='Source directory to validate')
    args = parser.parse_args()
    
    success = validate_data(args.src_dir)
    exit(0 if success else 1)