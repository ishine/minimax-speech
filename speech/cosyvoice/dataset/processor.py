# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw
import glob
import os
import json
import traceback
AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def individual_file_opener(data, mode='train', tts_data={}, token_latent_ratio=3):
    """Load data from individual files listed in files.txt
    
    Args:
        data: Iterable[{src}] where src is path to files.txt containing audio paths
        mode: 'train' or 'test'
        tts_data: Dict for TTS mode
        
    Yields:
        Dict with all required fields for training
    """
    for sample in data:
        assert 'src' in sample
        src = sample['src']
        
        # Load file list from files.txt
        file_list = []
        
        # Check if src is a files.txt file
        if src.endswith('.txt'):
            with open(src, 'r') as f:
                wav_files = [line.strip() for line in f if line.strip()]
            
            for wav_path in wav_files:
                # Skip empty lines or comments
                if not wav_path or wav_path.startswith('#'):
                    continue
                    
                # Verify wav file exists
                if not os.path.exists(wav_path):
                    logging.warning(f'Audio file not found: {wav_path}, skipping')
                    continue
                
                # Check if all required files exist
                txt_path = wav_path.replace('.wav', '.txt')
                token_path = wav_path.replace('.wav', '_fsq.pt')
                latent_path = wav_path.replace('.wav', '_latent2x.pt')
                
                if not os.path.exists(txt_path):
                    logging.warning(f'Text file not found for {wav_path}, skipping')
                    continue
                
                if not os.path.exists(token_path):
                    logging.warning(f'Token file not found for {wav_path}, skipping')
                    continue
                    
                if not os.path.exists(latent_path):
                    logging.warning(f'Latent file not found for {wav_path}, skipping')
                    continue
                
                # Extract metadata
                utt = os.path.basename(wav_path).replace('.wav', '')
                # Try to extract speaker from filename (assuming format: spk_*.wav)
                spk = utt.split('_')[0] if '_' in utt else 'default'
                
                file_info = {
                    'utt': utt,
                    'spk': spk,
                    'wav': wav_path,
                    'text_path': txt_path,
                    'token_path': token_path,
                    'latent_path': latent_path,
                }
                logging.info(f'file_info {file_info}')
                file_list.append(file_info)
        
        elif src.endswith('.json'):
            # Keep backward compatibility with JSON index files
            with open(src, 'r') as f:
                index_data = json.load(f)
                file_list = index_data.get('data', [])
        
        else:
            # Assume it's a directory for backward compatibility
            wav_files = glob.glob(os.path.join(src, '*/*/*wav'))
            if not wav_files:
                wav_files = glob.glob(os.path.join(src, '**/*.wav'), recursive=True)
            
            for wav_path in wav_files:
                txt_path = wav_path.replace('.wav', '.txt')
                token_path = wav_path.replace('.wav', '_fsq.pt')
                latent_path = wav_path.replace('.wav', '_latent2x.pt')
                
                if not os.path.exists(txt_path):
                    logging.warning(f'Text file not found for {wav_path}, skipping')
                    continue
                
                utt = os.path.basename(wav_path).replace('.wav', '')
                spk = utt.split('_')[0]
                
                file_info = {
                    'utt': utt,
                    'spk': spk,
                    'wav': wav_path,
                    'text_path': txt_path,
                    'token_path': token_path,
                    'latent_path': latent_path,
                }
                file_list.append(file_info)
        
        logging.info(f'Found {len(file_list)} valid audio files from {src}')
        
        # Process each file
        for file_info in file_list:
            try:
                # Read audio data
                with open(file_info['wav'], 'rb') as f:
                    audio_data = f.read()
                
                # Read text
                with open(file_info['text_path'], 'r', encoding='utf-8') as f:
                    text = ''.join(l.strip() for l in f.readlines())
                
                # Load speech token
                speech_token = torch.load(file_info['token_path'], map_location='cpu', weights_only=False)
                if isinstance(speech_token, torch.Tensor):
                    speech_token = speech_token.tolist()

                # Load speech latent
                speech_latent = torch.load(file_info['latent_path'], map_location='cpu', weights_only=False)
                speech_latent = speech_latent['z'].transpose(0, 1)

                if token_latent_ratio != 0:
                    # trim to align speech_token and speech_feat
                    print('before algin speech_latent: ', speech_latent.shape)
                    token_len = int(min(speech_latent.shape[0] / token_latent_ratio, len(speech_token)))
                    speech_latent = speech_latent[:token_latent_ratio * token_len]
                    speech_token = speech_token[:token_len]
                    print('after algin speech_latent: ', speech_latent.shape)

                # Build sample dict
                sample_dict = {
                    'utt': file_info['utt'],
                    'spk': file_info['spk'],
                    'audio_data': audio_data,
                    'text': text,
                    'text_token': [],  # Will be filled by tokenize processor
                    'speech_token': speech_token,
                    'wav': file_info['wav'],  # Keep original path for reference
                    'speech_latent': speech_latent,
                }
                
                # Copy over any additional fields from the original sample
                for key, value in sample.items():
                    if key not in sample_dict:
                        sample_dict[key] = value
                
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

def parquet_opener(data, mode='train', tts_data={}):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    sample.update(dict(df.loc[i]))
                    if mode == 'train':
                        # NOTE do not return sample directly, must initialize a new dict
                        yield {**sample}
                    else:
                        for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                            yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))
        sample['speech'] = sample['speech'].mean(dim=0, keepdim=True)
        del sample['audio_data']
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['speech'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['text_token']) < token_min_length:
            continue
        if len(sample['text_token']) > token_max_length:
            continue
        if len(sample['speech_token']) == 0:
            continue
        if 'reject_speech_token' in sample and len(sample['reject_speech_token']) == 0:
            continue
        if num_frames != 0:
            if len(sample['text_token']) / num_frames < min_output_input_ratio:
                print('continue text_token')
                continue
            if len(sample['text_token']) / num_frames > max_output_input_ratio:
                print('continue text_token')
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                print('continue sample_rate')
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
        else:
            waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)
        sample['speech'] = waveform
        yield sample


def extract_reference_mel_from_speech(
    data,
    feat_extractor,
    min_length=2.0,
    max_length=6.0,
    num_crops=2,  # Multiple random crops from same utterance
    training=True,
    sample_rate=24000,
    mode='train'
):
    """
    Extract mel spectrograms from current speech waveform with random cropping.
    This creates multiple random crops from the same utterance for training diversity.
    """
    for sample in data:
        # Use the current speech waveform
        waveform = sample['speech']  # [1, T]
        speech_length = waveform.shape[1]
        
        # Convert time to samples
        min_samples = int(min_length * sample_rate)
        max_samples = int(max_length * sample_rate)
        
        reference_mels = []
        reference_mel_lengths = []
        
        # Skip if utterance is too short
        if speech_length < min_samples:
            logging.warning(f"Speech for {sample['utt']} is too short ({speech_length/sample_rate:.2f}s)")
            sample['reference_mels'] = []
            sample['reference_mel_lengths'] = []
            sample['num_references'] = 0
            print('continue num_references')
            yield sample
            continue
        
        # Generate multiple crops from the same utterance
        crops_to_generate = num_crops if training else 1
        
        for i in range(crops_to_generate):
            if training and speech_length > max_samples:
                # Random crop during training
                crop_length = random.randint(min_samples, min(max_samples, speech_length))
                start_idx = random.randint(0, speech_length - crop_length)
                audio_segment = waveform[:, start_idx:start_idx + crop_length]
            elif speech_length > max_samples:
                # Center crop during inference
                start_idx = (speech_length - max_samples) // 2
                audio_segment = waveform[:, start_idx:start_idx + max_samples]
            else:
                # Use full audio if shorter than max_length
                audio_segment = waveform
                # For training, if we need multiple crops but audio is short,
                # we can add slight variations
                if training and i > 0:
                    # Add very slight noise for variation
                    noise = torch.randn_like(audio_segment) * 0.001
                    audio_segment = audio_segment + noise
            
            # Normalize audio segment
            max_val = audio_segment.abs().max()
            if max_val > 0:
                audio_segment = audio_segment / max_val
            
            # Extract mel spectrogram
            mel = feat_extractor(audio_segment).squeeze(0)  # Remove batch dim [C, T]
            reference_mels.append(mel)
            reference_mel_lengths.append(mel.shape[1])
        
        sample['reference_mels'] = reference_mels
        sample['reference_mel_lengths'] = reference_mel_lengths
        sample['num_references'] = len(reference_mels)
        
        yield sample


def compute_fbank(data,
                  feat_extractor,
                  token_mel_ratio=0,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        feat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        # if token_mel_ratio != 0:
        #     pass
            # trim to align speech_token and speech_feat
            # token_len = int(min(feat.shape[0] / token_mel_ratio, sample["speech_token"].shape[0]))
            # feat = feat[:token_mel_ratio * token_len]
            # sample["speech_token"] = sample["speech_token"][:token_len]
        sample['speech_mel'] = feat
        # print('feat shape, ', feat.shape)
        yield sample


def tokenize(data, get_tokenizer, allowed_special, mode='train'):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_latent'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['speech_latent'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_latent' in sample
        assert isinstance(sample['speech_latent'], torch.Tensor)
        new_sample_frames = sample['speech_latent'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))

def padding(data, mode='train', gan=False, dpo=False, use_speaker_encoder=False):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]
            use_speaker_encoder: Whether to prepare reference mels for speaker encoder

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_latent_len = torch.tensor([x['speech_latent'].size(0) for x in sample],  # Changed from size(1) to size(0)
                                       dtype=torch.int32)
        order = torch.argsort(speech_latent_len, descending=True)

        utts = [sample[i]['utt'] for i in order]
        speech = [sample[i]['speech'].squeeze(dim=0) for i in order]
        speech_len = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)
        speech = pad_sequence(speech, batch_first=True, padding_value=0)
        
        # Handle speech_token - check if it's already a tensor
        speech_token = []
        for i in order:
            if isinstance(sample[i]['speech_token'], torch.Tensor):
                speech_token.append(sample[i]['speech_token'])
            else:
                speech_token.append(torch.tensor(sample[i]['speech_token']))
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        speech_token = pad_sequence(speech_token,
                                    batch_first=True,
                                    padding_value=0)
        
        speech_latent = [sample[i]['speech_latent'] for i in order]

        speech_latent = pad_sequence(speech_latent,
                                   batch_first=True,
                                   padding_value=0)

        speech_mel = [sample[i]['speech_mel'] for i in order]
        speech_mel_len = torch.tensor([i.size(0) for i in speech_mel], dtype=torch.int32)
        speech_mel = pad_sequence(speech_mel,
                                   batch_first=True,
                                   padding_value=0)

        text = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        
        batch = {
            "utts": utts,
            "speech": speech,
            "speech_len": speech_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_mel": speech_mel,
            "speech_mel_len": speech_mel_len,
            "speech_latent": speech_latent,
            "speech_latent_len": speech_latent_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
        }
        
        # Handle reference mels for speaker encoder
        if use_speaker_encoder:
            # Collect all reference mels
            all_reference_mels = []
            all_reference_mel_lengths = []
            all_num_references = []
            
            for i in order:
                ref_mels = sample[i].get('reference_mels', [])
                ref_lengths = sample[i].get('reference_mel_lengths', [])
                num_refs = sample[i].get('num_references', 0)
                
                all_reference_mels.append(ref_mels)
                all_reference_mel_lengths.append(ref_lengths)
                all_num_references.append(num_refs)
            
            # Determine max number of references in batch
            max_num_refs = max(all_num_references) if all_num_references else 0
            
            if max_num_refs > 0:
                # Find dimensions
                batch_size = len(order)
                max_mel_length = 0
                mel_dim = 80  # default
                
                # Find max mel length and mel dimension
                for ref_mels in all_reference_mels:
                    for mel in ref_mels:
                        if isinstance(mel, torch.Tensor) and mel.numel() > 0:
                            max_mel_length = max(max_mel_length, mel.shape[1])
                            mel_dim = mel.shape[0]
                
                if max_mel_length > 0:
                    # Create padded tensor [B, N, C, T]
                    padded_reference_mels = torch.zeros(batch_size, max_num_refs, mel_dim, max_mel_length)
                    padded_reference_mel_lengths = torch.zeros(batch_size, max_num_refs, dtype=torch.int32)
                    reference_mel_masks = torch.zeros(batch_size, max_num_refs, max_mel_length)
                    
                    for b_idx, (ref_mels, ref_lengths) in enumerate(zip(all_reference_mels, all_reference_mel_lengths)):
                        for r_idx in range(min(len(ref_mels), max_num_refs)):
                            if r_idx < len(ref_mels) and isinstance(ref_mels[r_idx], torch.Tensor):
                                mel = ref_mels[r_idx]
                                length = ref_lengths[r_idx] if r_idx < len(ref_lengths) else mel.shape[1]
                                actual_length = min(length, mel.shape[1], max_mel_length)
                                
                                padded_reference_mels[b_idx, r_idx, :, :actual_length] = mel[:, :actual_length]
                                padded_reference_mel_lengths[b_idx, r_idx] = actual_length
                                reference_mel_masks[b_idx, r_idx, :actual_length] = 1.0
                    
                    batch['reference_mels'] = padded_reference_mels
                    batch['reference_mel_lengths'] = padded_reference_mel_lengths
                    batch['reference_mel_masks'] = reference_mel_masks
        
        if gan is True:
            # in gan train, we need pitch_feat
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            pitch_feat_len = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            pitch_feat = pad_sequence(pitch_feat,
                                      batch_first=True,
                                      padding_value=0)
            batch["pitch_feat"] = pitch_feat
            batch["pitch_feat_len"] = pitch_feat_len
        else:
            # only gan train needs speech, delete it to save memory
            del batch["speech"]
            del batch["speech_len"]
            
        if dpo is True:
            reject_speech_token = []
            for i in order:
                if isinstance(sample[i]['reject_speech_token'], torch.Tensor):
                    reject_speech_token.append(sample[i]['reject_speech_token'])
                else:
                    reject_speech_token.append(torch.tensor(sample[i]['reject_speech_token']))
            reject_speech_token_len = torch.tensor([i.size(0) for i in reject_speech_token], dtype=torch.int32)
            reject_speech_token = pad_sequence(reject_speech_token,
                                               batch_first=True,
                                               padding_value=0)
            batch['reject_speech_token'] = reject_speech_token
            batch['reject_speech_token_len'] = reject_speech_token_len
            
        yield batch