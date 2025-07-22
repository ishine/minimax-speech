# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Example Usage
cpu:

s3tokenizer --root_path /path/to/audio/files \
            --model speech_tokenizer_v1 \
            --device "cpu" \
            --batch_size 32

gpu:

torchrun --nproc_per_node=8 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    `which s3tokenizer` --root_path /path/to/audio/files \
                --model speech_tokenizer_v1 \
                --device "cuda" \
                --batch_size 32

"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import s3tokenizer


class AudioDataset(Dataset):

    def __init__(self, root_path, extensions=['.wav', '.flac', '.mp3'], 
                 use_cache=True, cache_file=None, max_workers=8):
        self.data = []
        
        # Define cache file path
        if cache_file is None:
            cache_file = Path(root_path) / '.audio_file_cache.pkl'
        else:
            cache_file = Path(cache_file)
        
        # Try to load from cache first
        if use_cache and cache_file.exists():
            import pickle
            print(f"Loading file list from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"Loaded {len(self.data)} files from cache")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}, scanning directory...")
        
        # Method 1: Use os.walk() which is typically faster than pathlib
        print(f"Scanning directory: {root_path}")
        print(f"Looking for extensions: {extensions}")
        
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def scan_directory(args):
            dirpath, extensions = args
            files = []
            try:
                with os.scandir(dirpath) as entries:
                    for entry in entries:
                        if entry.is_file() and any(entry.name.endswith(ext) for ext in extensions):
                            files.append(Path(entry.path))
            except PermissionError:
                pass
            return files
        
        # Collect all directories first
        all_dirs = [root_path]
        for dirpath, dirnames, _ in os.walk(root_path):
            all_dirs.extend(os.path.join(dirpath, d) for d in dirnames)
        
        # Process directories in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(scan_directory, (d, extensions)) for d in all_dirs]
            
            with tqdm(total=len(all_dirs), desc="Scanning directories") as pbar:
                for future in as_completed(futures):
                    self.data.extend(future.result())
                    pbar.update(1)
        
        # Sort for consistent ordering
        self.data.sort()
        
        if len(self.data) == 0:
            raise ValueError(f"No audio files found in {root_path}")
        
        print(f"Found {len(self.data)} audio files")
        
        # Save to cache
        if use_cache:
            try:
                import pickle
                print(f"Saving file list to cache: {cache_file}")
                cache_file.parent.mkdir(exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.data, f)
            except Exception as e:
                print(f"Failed to save cache: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        try:
            audio = s3tokenizer.load_audio(str(file_path))
            mel = s3tokenizer.log_mel_spectrogram(audio)
            return file_path, mel
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None


def collate_fn(batch):
    # Filter out None entries (failed files)
    batch = [item for item in batch if item[0] is not None]
    
    if len(batch) == 0:
        return [], None, None
    
    file_paths = [item[0] for item in batch]
    mels = [item[1] for item in batch]
    mels, mels_lens = s3tokenizer.padding(mels)
    return file_paths, mels, mels_lens


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print('Inference on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def get_args():
    parser = argparse.ArgumentParser(description='extract speech code')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        choices=[
                            "speech_tokenizer_v1", "speech_tokenizer_v1_25hz",
                            "speech_tokenizer_v2_25hz"
                        ],
                        help='model version')
    parser.add_argument('--root_path',
                        required=True,
                        type=str,
                        help='root directory containing audio files')
    parser.add_argument('--device',
                        required=True,
                        type=str,
                        choices=["cuda", "cpu"],
                        help='device for inference')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='batch size (per-device) for inference')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='workers for dataloader')
    parser.add_argument('--prefetch',
                        type=int,
                        default=5,
                        help='prefetch for dataloader')
    parser.add_argument('--extensions',
                        nargs='+',
                        default=['.wav', '.flac', '.mp3'],
                        help='audio file extensions to process')
    parser.add_argument('--use_cache',
                        action='store_true',
                        help='use cached file list to avoid re-scanning')
    parser.add_argument('--no_cache',
                        action='store_true',
                        help='force re-scan even if cache exists')
    parser.add_argument('--cache_file',
                        type=str,
                        default=None,
                        help='path to cache file (default: root_path/.audio_file_cache.pkl)')
    parser.add_argument('--scan_workers',
                        type=int,
                        default=8,
                        help='number of workers for directory scanning')
    parser.add_argument('--file_list',
                        type=str,
                        default=None,
                        help='path to pre-generated file list (one file per line)')
    parser.add_argument('--skip_existing',
                        action='store_true',
                        help='skip files that already have _fsq.pt output')
    args = parser.parse_args()
    return args


def save_tokens(file_path, codes, codes_len):
    """Save tokens as .pt file with _fsq suffix"""
    # Remove extension and add _fsq.pt
    output_path = file_path.with_suffix('').with_suffix('.pt')
    output_path = output_path.parent / f"{output_path.stem}_fsq.pt"
    
    # Extract only valid codes (up to codes_len)
    valid_codes = codes[:codes_len]
    
    # Save as tensor
    torch.save(valid_codes, output_path)
    
    return output_path


def main():
    args = get_args()

    if args.device == "cuda":
        assert (torch.cuda.is_available())
        world_size, local_rank, rank = init_distributed()
    else:
        world_size, local_rank, rank = 1, 0, 0

    device = torch.device(args.device)
    model = s3tokenizer.load_model(args.model).to(device)
    
    # Handle different data loading methods
    if args.file_list:
        # Option 3: Load from pre-generated file list
        print(f"Loading file list from: {args.file_list}")
        with open(args.file_list, 'r') as f:
            file_paths = [Path(line.strip()) for line in f if line.strip()]
        
        # Filter by extensions if specified
        if args.extensions:
            file_paths = [f for f in file_paths if any(str(f).endswith(ext) for ext in args.extensions)]
        
        # Create a simple dataset
        class FileListDataset(Dataset):
            def __init__(self, file_paths, skip_existing=False):
                self.data = []
                skipped_existing = 0
                for fp in file_paths:
                    if skip_existing:
                        output_path = fp.with_suffix('').with_suffix('.pt')
                        output_path = output_path.parent / f"{output_path.stem}_fsq.pt"
                        if output_path.exists():
                            skipped_existing += 1
                            continue
                    self.data.append(fp)
                print(f"Will process {len(self.data)} files")
                if skip_existing and skipped_existing > 0:
                    print(f"Skipped {skipped_existing} already processed files")
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                file_path = self.data[idx]
                try:
                    # Check if file exists
                    if not file_path.exists():
                        print(f"File not found: {file_path}")
                        return None, None
                    
                    # Check if it's a file (not directory)
                    if not file_path.is_file():
                        print(f"Not a file: {file_path}")
                        return None, None
                    
                    # Try to load audio
                    audio = s3tokenizer.load_audio(str(file_path))
                    mel = s3tokenizer.log_mel_spectrogram(audio)
                    return file_path, mel
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    return None, None
        
        dataset = FileListDataset(file_paths, skip_existing=args.skip_existing)
    else:
        # Use the enhanced AudioDataset with caching
        dataset = AudioDataset(
            args.root_path, 
            args.extensions,
            use_cache=not args.no_cache,
            cache_file=args.cache_file,
            max_workers=args.scan_workers
        )
        
        # Filter out existing files if requested
        if args.skip_existing:
            original_count = len(dataset.data)
            dataset.data = [
                fp for fp in dataset.data
                if not (fp.parent / f"{fp.stem}_fsq.pt").exists()
            ]
            print(f"Skipping {original_count - len(dataset.data)} already processed files")

    if args.device == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
        sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=rank)
    else:
        sampler = None

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch,
                            collate_fn=collate_fn)

    total_steps = len(dataset)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    processed_count = 0
    failed_count = 0
    failed_files = []
    
    for file_paths, mels, mels_lens in dataloader:
        # Skip empty batches (all files failed)
        if len(file_paths) == 0:
            continue
            
        codes, codes_lens = model(mels.to(device), mels_lens.to(device))
        
        # Process each file in the batch
        for i, file_path in enumerate(file_paths):
            try:
                code = codes[i]
                code_len = codes_lens[i].item()
                
                # Save tokens as .pt file
                output_path = save_tokens(file_path, code, code_len)
                
                if rank == 0 and processed_count < 10:  # Only show first 10 to avoid spam
                    tqdm.write(f"Saved: {file_path} -> {output_path}")
                
                processed_count += 1
            except Exception as e:
                failed_count += 1
                failed_files.append(str(file_path))
                if rank == 0:
                    tqdm.write(f"Failed to save {file_path}: {e}")
        
        if rank == 0:
            progress_bar.update(world_size * (len(file_paths) + failed_count))

    if rank == 0:
        progress_bar.close()
        print(f"\nProcessed {processed_count} files successfully on rank {rank}")
        if failed_count > 0:
            print(f"Failed to process {failed_count} files")
            
            # Save failed files list
            failed_list_path = Path(args.root_path if not args.file_list else ".") / "failed_files.txt"
            with open(failed_list_path, 'w') as f:
                for ff in failed_files:
                    f.write(f"{ff}\n")
            print(f"Failed files saved to: {failed_list_path}")
    
    if args.device == "cuda":
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()