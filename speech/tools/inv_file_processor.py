# Add this to your processor.py file or create a new file

import logging
import json
import torch
import glob
import os
from pathlib import Path

def individual_file_opener(data, mode='train', tts_data={}):
    """Load data from individual files instead of parquet
    
    Args:
        data: Iterable[{src}] where src is either:
            - Path to index JSON file
            - Directory path containing wav files
        mode: 'train' or 'test'
        tts_data: Dict for TTS mode
        
    Yields:
        Dict with all required fields for training
    """
    for sample in data:
        src = sample['src']
        
        # Check if src is a JSON index file or a directory
        if src.endswith('.json'):
            # Load from index file
            with open(src, 'r') as f:
                index_data = json.load(f)
                file_list = index_data.get('data', [])
        else:
            # Find all wav files in directory
            wav_files = glob.glob(os.path.join(src, '*/*/*wav'))
            file_list = []
            for wav_path in wav_files:
                # Check if all required files exist
                txt_path = wav_path.replace('.wav', '.normalized.txt')
                embedding_path = wav_path.replace('.wav', '_embedding.pt')
                token_path = wav_path.replace('.wav', '_tokens.pt')
                
                if not all(os.path.exists(p) for p in [txt_path, embedding_path, token_path]):
                    logging.warning(f'Missing files for {wav_path}, skipping')
                    continue
                
                # Extract metadata
                utt = os.path.basename(wav_path).replace('.wav', '')
                spk = utt.split('_')[0]
                
                file_list.append({
                    'utt': utt,
                    'spk': spk,
                    'wav': wav_path,
                    'text_path': txt_path,
                    'embedding_path': embedding_path,
                    'token_path': token_path,
                    'spk_embedding_path': os.path.join(os.path.dirname(src), f"spk_embeddings/{spk}_embedding.pt")
                })
        
        # Process each file
        for file_info in file_list:
            try:
                # Read audio data
                with open(file_info['wav'], 'rb') as f:
                    audio_data = f.read()
                
                # Read text
                with open(file_info['text_path'], 'r') as f:
                    text = ''.join(l.strip() for l in f.readlines())
                
                # Load embeddings
                utt_embedding = torch.load(file_info['embedding_path']).tolist()
                speech_token = torch.load(file_info['token_path'])
                
                # Load speaker embedding
                if os.path.exists(file_info['spk_embedding_path']):
                    spk_embedding = torch.load(file_info['spk_embedding_path']).tolist()
                else:
                    logging.warning(f"Speaker embedding not found: {file_info['spk_embedding_path']}")
                    spk_embedding = utt_embedding  # Fallback to utterance embedding
                
                # Build sample dict
                sample_dict = {
                    'utt': file_info['utt'],
                    'spk': file_info['spk'],
                    'audio_data': audio_data,
                    'text': text,
                    'text_token': [],  # Will be filled by tokenize processor
                    'utt_embedding': utt_embedding,
                    'spk_embedding': spk_embedding,
                    'speech_token': speech_token,
                    'wav': file_info['wav'],  # Keep original path for reference
                }
                
                # Merge with original sample data
                sample_dict.update(sample)
                
                if mode == 'train':
                    yield sample_dict
                else:
                    # For TTS mode
                    if file_info['utt'] in tts_data:
                        for index, tts_text in enumerate(tts_data[file_info['utt']]):
                            yield {**sample_dict, 'tts_index': index, 'tts_text': tts_text}
                    else:
                        yield sample_dict
                        
            except Exception as ex:
                logging.warning(f'Failed to process {file_info["wav"]}: {ex}')