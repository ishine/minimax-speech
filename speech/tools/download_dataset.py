import os
import io
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Queue, Process
from threading import Lock
from queue import Empty
import logging
from tqdm import tqdm
from functools import partial

import numpy as np
import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global settings
CHUNK_SIZE = 100  # Process this many samples before showing progress
MAX_WORKERS = cpu_count() - 1  # Leave one CPU free
PREFETCH_SIZE = 50  # How many samples to prefetch

def process_single_sample(data, root_save_path):
    """Process a single sample from the dataset"""
    try:
        metadata = data['json']
        
        # Prepare paths
        out_wav_path = os.path.join(root_save_path, metadata['wav'].replace('/mp3', '').replace('.mp3', '.wav'))
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        out_text_path = out_wav_path.replace('.wav', '.txt')
        # check if the files are existsed
        if os.path.exists(out_wav_path) and os.path.exists(out_text_path):
            return metadata['id'], True, None
        
        # Save text
        with open(out_text_path, 'w') as f:
            f.write(metadata['text'])
        
        # Decode and save audio
        raw_bytes = data['mp3']._hf_encoded['bytes']
        audio_tensor, sample_rate = torchaudio.load(io.BytesIO(raw_bytes), format='mp3')
        audio_array = audio_tensor.squeeze().numpy()
        sf.write(out_wav_path, audio_array, sample_rate)
        
        return metadata['id'], True, None
    except Exception as e:
        return data.get('json', {}).get('id', 'unknown'), False, str(e)

def process_batch(batch_data, root_save_path):
    """Process a batch of samples"""
    results = []
    for data in batch_data:
        result = process_single_sample(data, root_save_path)
        results.append(result)
    return results

class ParallelDatasetProcessor:
    """Main class for parallel processing of the dataset"""
    
    def __init__(self, language, root_save_path, num_workers=None):
        self.language = language
        self.root_save_path = root_save_path
        self.num_workers = num_workers or MAX_WORKERS
        self.processed_count = 0
        self.error_count = 0
        self.lock = Lock()
        
    def process_with_multiprocessing(self):
        """Process dataset using multiprocessing (fastest for CPU-bound tasks)"""
        logger.info(f"Starting multiprocessing with {self.num_workers} workers")
        
        # Load dataset
        path = f"Emilia/{self.language.upper()}/*.tar"
        dataset = load_dataset(
            "amphion/Emilia-Dataset", 
            data_files={self.language: path}, 
            split=self.language, 
            streaming=True
        )
        
        # Create process pool
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit jobs in batches
            futures = []
            batch = []
            
            # Progress bar
            pbar = tqdm(desc="Processing samples", unit="samples")
            
            for data in dataset:
                batch.append(data)
                
                if len(batch) >= CHUNK_SIZE:
                    # Submit batch for processing
                    future = executor.submit(process_batch, batch, self.root_save_path)
                    futures.append(future)
                    batch = []
                    
                    # Process completed futures
                    self._process_completed_futures(futures, pbar, max_pending=self.num_workers * 2)
            
            # Submit remaining batch
            if batch:
                future = executor.submit(process_batch, batch, self.root_save_path)
                futures.append(future)
            
            # Wait for all remaining futures
            for future in as_completed(futures):
                results = future.result()
                for sample_id, success, error in results:
                    if success:
                        self.processed_count += 1
                    else:
                        self.error_count += 1
                        logger.error(f"Error processing {sample_id}: {error}")
                    pbar.update(1)
            
            pbar.close()
    
    def process_with_threading(self):
        """Process dataset using threading (good for I/O-bound tasks)"""
        logger.info(f"Starting threading with {self.num_workers} workers")
        
        # Load dataset
        path = f"Emilia/{self.language.upper()}/*.tar"
        dataset = load_dataset(
            "amphion/Emilia-Dataset", 
            data_files={self.language: path}, 
            split=self.language, 
            streaming=True
        )
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=self.num_workers * 2) as executor:
            futures = []
            pbar = tqdm(desc="Processing samples", unit="samples")
            
            for data in dataset:
                # Submit individual samples
                future = executor.submit(process_single_sample, data, self.root_save_path)
                futures.append((future, data.get('json', {}).get('id', 'unknown')))
                
                # Process completed futures
                if len(futures) >= self.num_workers * 4:
                    completed = []
                    for i, (future, sample_id) in enumerate(futures):
                        if future.done():
                            try:
                                _, success, error = future.result()
                                if success:
                                    self.processed_count += 1
                                else:
                                    self.error_count += 1
                                    logger.error(f"Error processing {sample_id}: {error}")
                                pbar.update(1)
                                completed.append(i)
                            except Exception as e:
                                logger.error(f"Exception processing {sample_id}: {e}")
                                self.error_count += 1
                                pbar.update(1)
                                completed.append(i)
                    
                    # Remove completed futures
                    for i in reversed(completed):
                        futures.pop(i)
            
            # Wait for remaining futures
            for future, sample_id in futures:
                try:
                    _, success, error = future.result()
                    if success:
                        self.processed_count += 1
                    else:
                        self.error_count += 1
                        logger.error(f"Error processing {sample_id}: {error}")
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Exception processing {sample_id}: {e}")
                    self.error_count += 1
                    pbar.update(1)
            
            pbar.close()
    
    def process_with_producer_consumer(self):
        """Process dataset using producer-consumer pattern"""
        logger.info(f"Starting producer-consumer with {self.num_workers} workers")
        
        # Create queues
        work_queue = Queue(maxsize=PREFETCH_SIZE)
        result_queue = Queue()
        
        # Start producer
        producer = Process(target=self._producer, args=(work_queue,))
        producer.start()
        
        # Start consumers
        consumers = []
        for i in range(self.num_workers):
            consumer = Process(target=self._consumer, args=(work_queue, result_queue, i))
            consumer.start()
            consumers.append(consumer)
        
        # Start result processor
        result_processor = Process(target=self._result_processor, args=(result_queue,))
        result_processor.start()
        
        # Wait for completion
        producer.join()
        
        # Signal consumers to stop
        for _ in range(self.num_workers):
            work_queue.put(None)
        
        for consumer in consumers:
            consumer.join()
        
        # Signal result processor to stop
        result_queue.put(None)
        result_processor.join()
    
    def _producer(self, work_queue):
        """Producer process that reads from dataset"""
        path = f"Emilia/{self.language.upper()}/*.tar"
        dataset = load_dataset(
            "amphion/Emilia-Dataset", 
            data_files={self.language: path}, 
            split=self.language, 
            streaming=True
        )
        
        for data in dataset:
            work_queue.put(data)
    
    def _consumer(self, work_queue, result_queue, worker_id):
        """Consumer process that processes samples"""
        while True:
            try:
                data = work_queue.get(timeout=1)
                if data is None:
                    break
                
                result = process_single_sample(data, self.root_save_path)
                result_queue.put(result)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def _result_processor(self, result_queue):
        """Process results and update progress"""
        pbar = tqdm(desc="Processing samples", unit="samples")
        
        while True:
            try:
                result = result_queue.get(timeout=1)
                if result is None:
                    break
                
                sample_id, success, error = result
                if success:
                    self.processed_count += 1
                else:
                    self.error_count += 1
                    logger.error(f"Error processing {sample_id}: {error}")
                pbar.update(1)
            except Empty:
                continue
        
        pbar.close()
    
    def _process_completed_futures(self, futures, pbar, max_pending):
        """Process completed futures to avoid memory buildup"""
        while len(futures) > max_pending:
            # Wait for at least one to complete
            completed_futures = []
            for i, future in enumerate(futures):
                if future.done():
                    results = future.result()
                    for sample_id, success, error in results:
                        if success:
                            self.processed_count += 1
                        else:
                            self.error_count += 1
                            logger.error(f"Error processing {sample_id}: {error}")
                        pbar.update(1)
                    completed_futures.append(i)
            
            # Remove completed futures
            for i in reversed(completed_futures):
                futures.pop(i)
            
            if not completed_futures:
                # If nothing completed, wait a bit
                time.sleep(0.1)

def main():
    """Main function with different processing options"""
    language = "en"
    root_save_path = f'/data/emilia/{language}'
    os.makedirs(root_save_path, exist_ok=True)
    
    # Choose processing method
    processor = ParallelDatasetProcessor(language, root_save_path)
    
    # Method 1: Multiprocessing (recommended for CPU-bound tasks)
    logger.info("Starting parallel processing...")
    start_time = time.time()
    
    # You can choose one of these methods:
    processor.process_with_multiprocessing()  # Fastest for this use case
    # processor.process_with_threading()      # Good for I/O heavy tasks
    # processor.process_with_producer_consumer()  # Good for memory efficiency
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully processed: {processor.processed_count} samples")
    logger.info(f"Errors: {processor.error_count} samples")

# Simplified version for quick use
def simple_parallel_process():
    """Simplified parallel processing function"""
    from joblib import Parallel, delayed
    
    language = "en"
    root_save_path = f'/data/emilia/{language}'
    os.makedirs(root_save_path, exist_ok=True)
    
    # Load dataset
    path = f"Emilia/{language.upper()}/*.tar"
    dataset = load_dataset(
        "amphion/Emilia-Dataset", 
        data_files={language: path}, 
        split=language, 
        streaming=True
    )
    
    # Collect samples in batches
    batch_size = 1000
    batch = []
    
    def process_batch_simple(batch_data):
        for data in batch_data:
            process_single_sample(data, root_save_path)
    
    # Process in parallel batches
    for i, data in enumerate(tqdm(dataset, desc="Loading samples")):
        batch.append(data)
        
        if len(batch) >= batch_size:
            # Process batch in parallel
            Parallel(n_jobs=-1, backend='multiprocessing')(
                delayed(process_single_sample)(sample, root_save_path) 
                for sample in batch
            )
            batch = []
    
    # Process remaining batch
    if batch:
        Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(process_single_sample)(sample, root_save_path) 
            for sample in batch
        )

if __name__ == "__main__":
    main()