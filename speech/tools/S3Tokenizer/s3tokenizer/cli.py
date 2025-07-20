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
            --model speech_tokenizer_v2_25hz \
            --device "cpu" \
            --batch_size 32

gpu:

torchrun --nproc_per_node=1 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    `which s3tokenizer` --root_path /data/dataset \
                --model speech_tokenizer_v2_25hz \
                --device "cuda" \
                --batch_size 64

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

    def __init__(self, root_path, extensions=['.wav', '.flac', '.mp3']):
        self.data = []
        
        # Recursively find all audio files
        root = Path(root_path)
        for ext in extensions:
            self.data.extend(root.rglob(f'*{ext}'))
        
        # Sort for consistent ordering
        self.data.sort()
        
        if len(self.data) == 0:
            raise ValueError(f"No audio files found in {root_path}")
        
        print(f"Found {len(self.data)} audio files")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        audio = s3tokenizer.load_audio(str(file_path))
        mel = s3tokenizer.log_mel_spectrogram(audio)
        return file_path, mel


def collate_fn(batch):
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
    args = parser.parse_args()
    return args


def save_tokens(file_path, codes, codes_len):
    """Save tokens as .pt file with _fsq suffix"""
    # Remove extension and add _fsq.pt
    output_path = file_path.with_suffix('').with_suffix('.pt')
    output_path = output_path.parent / f"{output_path.stem}_fsq.pt"
    
    # Extract only valid codes (up to codes_len)
    valid_codes = codes[:codes_len]
    # convert valid codes to list
    valid_codes = valid_codes.tolist()
    
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
    dataset = AudioDataset(args.root_path, args.extensions)

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
    for file_paths, mels, mels_lens in dataloader:
        codes, codes_lens = model(mels.to(device), mels_lens.to(device))
        
        # Process each file in the batch
        for i, file_path in enumerate(file_paths):
            code = codes[i]
            code_len = codes_lens[i].item()
            
            # Save tokens as .pt file
            output_path = save_tokens(file_path, code, code_len)
            
            if rank == 0:
                tqdm.write(f"Saved: {file_path} -> {output_path}")
        
        processed_count += len(file_paths)
        
        if rank == 0:
            progress_bar.update(world_size * len(file_paths))

    if rank == 0:
        progress_bar.close()
        print(f"\nProcessed {processed_count} files on rank {rank}")
    
    if args.device == "cuda":
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()