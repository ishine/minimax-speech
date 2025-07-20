import os
import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing
from typing import Dict, Any, Optional

from datasets import load_dataset
import soundfile as sf
import torchaudio
from tqdm import tqdm


def process_single_item(data: Dict[str, Any], root_save_path: str) -> Optional[str]:
    """
    Process a single audio item: extract audio, convert, and save with text.
    
    Args:
        data: Dictionary containing audio data and text
        root_save_path: Root directory for saving files
        
    Returns:
        Audio file path if successful, None if error
    """
    try:
        # Extract audio data
        raw_bytes = data['audio']._hf_encoded['bytes']
        audio_name = data['audio']._hf_encoded['path']
        # Prepare output paths
        out_wav_path = os.path.join(root_save_path, audio_name)
        out_text_path = out_wav_path.replace('.wav', '.txt')

        if os.path.exists(out_wav_path) and os.path.exists(out_text_path):
            # print(f'skip {out_wav_path}')
            return out_wav_path
        
        text = data['text']
        
        # Load audio from bytes
        bytes_io = io.BytesIO(raw_bytes)
        audio_tensor, sample_rate = torchaudio.load(bytes_io)
        audio_array = audio_tensor.squeeze().numpy()
        
        
        
        # Create directory if needed
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        
        # Save audio and text
        sf.write(out_wav_path, audio_array, sample_rate)
        with open(out_text_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return out_wav_path
        
    except Exception as e:
        print(f"Error processing {data.get('audio', {})._hf_encoded.get('path', 'unknown')}: {e}")
        return None


def process_dataset_parallel(
    dataset_name: str = "capleaf/viVoice",
    root_save_path: str = '/mnt/nvme-temp/vivoice',
    max_workers: Optional[int] = None,
    batch_size: int = 100,
    limit: Optional[int] = None,
    use_threads: bool = False
):
    """
    Process dataset in parallel with progress tracking.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        root_save_path: Root directory for saving files
        max_workers: Maximum number of parallel workers (None for CPU count)
        batch_size: Number of items to process in each batch
        limit: Maximum number of items to process (None for all)
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor
    """
    # Load dataset
    ds = load_dataset(dataset_name, streaming=True)
    
    # Set up executor
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    # Process each split
    for mode in ds:
        print(f"\nProcessing '{mode}' split...")
        
        # Create partial function with root_save_path
        process_func = partial(process_single_item, root_save_path=root_save_path)
        
        # Collect items in batches for better progress tracking
        batch = []
        processed_count = 0
        
        with Executor(max_workers=max_workers) as executor:
            # Create progress bar
            pbar = tqdm(desc=f"Processing {mode}", unit="files")
            
            for idx, data in enumerate(ds[mode]):
                batch.append(data)
                
                # Process batch when full or at limit
                if len(batch) >= batch_size or (limit and idx + 1 >= limit):
                    # Submit batch for processing
                    futures = [executor.submit(process_func, item) for item in batch]
                    
                    # Wait for completion and update progress
                    for future in futures:
                        result = future.result()
                        if result:
                            processed_count += 1
                        pbar.update(1)
                    
                    batch = []
                
                # Stop if limit reached
                if limit and idx + 1 >= limit:
                    break
            
            # Process remaining items
            if batch:
                futures = [executor.submit(process_func, item) for item in batch]
                for future in futures:
                    result = future.result()
                    if result:
                        processed_count += 1
                    pbar.update(1)
            
            pbar.close()
        
        print(f"Completed {mode}: {processed_count} files processed successfully")


def process_dataset_streaming(
    dataset_name: str = "capleaf/viVoice",
    root_save_path: str = '/data/vivoice',
    max_workers: Optional[int] = None,
    limit: Optional[int] = None
):
    """
    Alternative approach using streaming with thread pool for I/O bound operations.
    More memory efficient for very large datasets.
    """
    ds = load_dataset(dataset_name, streaming=True)
    
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() * 4)  # More threads for I/O
    
    for mode in ds:
        print(f"\nProcessing '{mode}' split...")
        
        process_func = partial(process_single_item, root_save_path=root_save_path)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit items as they come from the stream
            futures = []
            
            for idx, data in enumerate(tqdm(ds[mode], desc=f"Submitting {mode}")):
                future = executor.submit(process_func, data)
                futures.append(future)
                
                # Limit number of pending futures to control memory usage
                if len(futures) >= max_workers * 2:
                    # Wait for some to complete
                    for f in futures[:max_workers]:
                        f.result()
                    futures = futures[max_workers:]
                
                if limit and idx + 1 >= limit:
                    break
            
            # Wait for remaining futures
            for future in tqdm(futures, desc="Finishing"):
                future.result()


if __name__ == "__main__":
    # Example usage - choose one approach:
    
    # Approach 1: Process with multiprocessing (good for CPU-bound operations)
    process_dataset_parallel(
        dataset_name="capleaf/viVoice",
        root_save_path='/data/vivoice',
        max_workers=24,  # Adjust based on your system
        batch_size=100,
        limit=None,  # Remove to process all data
        use_threads=False
    )
    
    # Approach 2: Process with threading (good for I/O-bound operations)
    # process_dataset_parallel(
    #     dataset_name="capleaf/viVoice",
    #     root_save_path='/mnt/nvme-temp/vivoice',
    #     max_workers=16,  # Can use more threads for I/O
    #     batch_size=100,
    #     limit=None,
    #     use_threads=True
    # )
    
    # Approach 3: Streaming approach (most memory efficient)
    # process_dataset_streaming(
    #     dataset_name="capleaf/viVoice",
    #     root_save_path='/mnt/nvme-temp/vivoice',
    #     max_workers=16,
    #     limit=None
