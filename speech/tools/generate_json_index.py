#!/usr/bin/env python3
"""
Generate JSON index file for dataset
This creates a JSON file with all valid audio files and their metadata
"""

import os
import json
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from datetime import datetime

def validate_file_set(wav_path):
    """Check if all required files exist for a given wav file"""
    txt_path = wav_path.replace('.wav', '.txt')
    token_path = wav_path.replace('.wav', '_fsq.pt')
    latent_path = wav_path.replace('.wav', '_latent.pt')
    
    # Check all files exist
    if not all(os.path.exists(p) for p in [wav_path, txt_path, token_path, latent_path]):
        return None
    
    # Get file sizes for validation
    try:
        wav_size = os.path.getsize(wav_path)
        txt_size = os.path.getsize(txt_path)
        token_size = os.path.getsize(token_path)
        latent_size = os.path.getsize(latent_path)
        
        # Skip if any file is empty
        if any(size == 0 for size in [wav_size, txt_size, token_size, latent_size]):
            return None
        
        # Extract metadata
        utt = os.path.basename(wav_path).replace('.wav', '')
        spk = utt.split('_')[0] if '_' in utt else 'default'
        
        # Get file modification time
        mtime = os.path.getmtime(wav_path)
        
        return {
            'utt': utt,
            'spk': spk,
            'wav': wav_path,
            'text_path': txt_path,
            'token_path': token_path,
            'latent_path': latent_path,
            'wav_size': wav_size,
            'txt_size': txt_size,
            'token_size': token_size,
            'latent_size': latent_size,
            'mtime': mtime,
        }
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def process_directory(directory, pattern='**/*.wav'):
    """Process a directory and find all valid audio files"""
    print(f"Scanning directory: {directory}")
    wav_files = glob.glob(os.path.join(directory, pattern), recursive=True)
    print(f"Found {len(wav_files)} wav files")
    
    valid_files = []
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(validate_file_set, wav_path) for wav_path in wav_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating files"):
            result = future.result()
            if result is not None:
                valid_files.append(result)
    
    return valid_files

def process_files_txt(files_txt):
    """Process files from a text file list"""
    print(f"Reading file list from: {files_txt}")
    
    with open(files_txt, 'r') as f:
        wav_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Found {len(wav_files)} files in list")
    
    valid_files = []
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(validate_file_set, wav_path) for wav_path in wav_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating files"):
            result = future.result()
            if result is not None:
                valid_files.append(result)
    
    return valid_files

def generate_statistics(file_list):
    """Generate statistics about the dataset"""
    stats = {
        'total_files': len(file_list),
        'total_size_gb': sum(f['wav_size'] + f['txt_size'] + f['token_size'] + f['latent_size'] 
                           for f in file_list) / (1024**3),
        'speakers': {},
        'file_sizes': {
            'wav_total_gb': sum(f['wav_size'] for f in file_list) / (1024**3),
            'txt_total_mb': sum(f['txt_size'] for f in file_list) / (1024**2),
            'token_total_gb': sum(f['token_size'] for f in file_list) / (1024**3),
            'latent_total_gb': sum(f['latent_size'] for f in file_list) / (1024**3),
        }
    }
    
    # Count files per speaker
    for f in file_list:
        spk = f['spk']
        if spk not in stats['speakers']:
            stats['speakers'][spk] = 0
        stats['speakers'][spk] += 1
    
    stats['num_speakers'] = len(stats['speakers'])
    
    return stats

def generate_json_index(input_paths, output_file, split_ratio=None):
    """
    Generate JSON index file from input paths
    
    Args:
        input_paths: List of paths (directories or files.txt)
        output_file: Output JSON file path
        split_ratio: Optional tuple (train_ratio, val_ratio, test_ratio)
    """
    all_files = []
    
    # Process each input path
    for path in input_paths:
        if os.path.isdir(path):
            files = process_directory(path)
        elif path.endswith('.txt'):
            files = process_files_txt(path)
        else:
            print(f"Warning: Skipping unknown input type: {path}")
            continue
        
        all_files.extend(files)
    
    # Remove duplicates based on utterance ID
    unique_files = {}
    for f in all_files:
        utt = f['utt']
        if utt not in unique_files:
            unique_files[utt] = f
        else:
            # Keep the one with newer modification time
            if f['mtime'] > unique_files[utt]['mtime']:
                unique_files[utt] = f
    
    file_list = list(unique_files.values())
    
    # Sort by utterance ID for consistency
    file_list.sort(key=lambda x: x['utt'])
    
    print(f"\nTotal unique files: {len(file_list)}")
    
    # Generate statistics
    stats = generate_statistics(file_list)
    
    # Create index
    index = {
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'statistics': stats,
        'data': file_list
    }
    
    # Optional: Create train/val/test splits
    if split_ratio:
        import random
        random.seed(42)  # For reproducibility
        
        train_ratio, val_ratio, test_ratio = split_ratio
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Split ratios must sum to 1"
        
        # Shuffle for random split
        shuffled = file_list.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }
        
        # Save separate files for each split
        base_name = output_file.replace('.json', '')
        
        for split_name, split_data in splits.items():
            split_index = {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'split': split_name,
                'statistics': generate_statistics(split_data),
                'data': split_data
            }
            
            split_file = f"{base_name}_{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump(split_index, f, indent=2)
            
            print(f"Saved {split_name} split: {split_file} ({len(split_data)} files)")
    
    # Save main index
    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nSaved index to: {output_file}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_gb']:.2f} GB")
    print(f"Number of speakers: {stats['num_speakers']}")

def main():
    parser = argparse.ArgumentParser(description="Generate JSON index for dataset")
    parser.add_argument('--input', nargs='+', required=True,
                      help='Input paths (directories or files.txt)')
    parser.add_argument('--output', default='dataset_index.json',
                      help='Output JSON file (default: dataset_index.json)')
    parser.add_argument('--split', nargs=3, type=float, metavar=('TRAIN', 'VAL', 'TEST'),
                      help='Create train/val/test splits (e.g., --split 0.8 0.1 0.1)')
    
    args = parser.parse_args()
    
    # Validate split ratios if provided
    split_ratio = None
    if args.split:
        split_ratio = tuple(args.split)
        if abs(sum(split_ratio) - 1.0) > 0.001:
            parser.error("Split ratios must sum to 1.0")
    
    generate_json_index(args.input, args.output, split_ratio)

if __name__ == "__main__":
    main()

# Example usage:
# python generate_json_index.py --input /data/dataset/emilia /data/dataset/vivoice --output dataset_index.json
# python generate_json_index.py --input train_files.txt --output train_index.json
# python generate_json_index.py --input /data/dataset/emilia --output dataset_index.json --split 0.8 0.1 0.1