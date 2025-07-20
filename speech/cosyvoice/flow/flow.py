# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
from ast import List
import logging
import random
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        encoder: torch.nn.Module = None,
        length_regulator: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                    "use_immiscible": True,
                    "immiscible_k": 8,
                    "use_contrastive_fm": False,
                    "contrastive_lambda": 0.05
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        print("token_len values: ", token_len)
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        # NOTE this is unnecessary, feat/h already same shape
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
        )
        return {"loss": loss}

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        flow_cache,
    ):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech token and prompt speech token
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = (
            torch.concat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(
            token_len2 / self.input_frame_rate * 22050 / 256
        )
        h, h_lengths = self.length_regulator.inference(
            h[:, :token_len1],
            h[:, token_len1:],
            mel_len1,
            mel_len2,
            self.input_frame_rate,
        )

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        use_speaker_encoder: bool = False,  # Add this
        freeze_speaker_encoder: bool = False,  # Add this
        max_conditioning_inputs: int = 2,  # Add this
        speaker_encoder_path: str = None,
        encoder: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                    "use_immiscible": True,
                    "immiscible_k": 8,
                    "use_contrastive_fm": True,
                    "contrastive_lambda": 0.05
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.use_speaker_encoder = use_speaker_encoder
        # Speaker encoder setup
        if use_speaker_encoder:
            from cosyvoice.llm.llm import LearnableSpeakerEncoder
            self.speaker_encoder = LearnableSpeakerEncoder(
                mel_dim=80,
                model_dim=512,
                output_dim=spk_embed_dim,
                num_blocks=6,
                num_heads=8,
            )
            
            # Load speaker encoder weights from LLM checkpoint if provided
        if speaker_encoder_path is not None:
            logging.info(f"Loading speaker encoder from {speaker_encoder_path}")
            checkpoint = torch.load(speaker_encoder_path, map_location='cpu')
            
            # Debug: print checkpoint structure
            print(f'Checkpoint keys: {checkpoint.keys()}')
            
            # Extract speaker encoder weights
            speaker_encoder_state = {}
            
            # Check if checkpoint has 'state_dict' key or direct model weights
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Direct model weights (based on your save function)
                state_dict = {k: v for k, v in checkpoint.items() if not k in ['epoch', 'step']}
            
            # Extract speaker encoder weights
            for key, value in state_dict.items():
                if 'speaker_encoder.' in key:
                    # Remove module. prefix if exists (from DDP)
                    new_key = key.replace('module.', '')
                    # Remove speaker_encoder. prefix to match the local module
                    new_key = new_key.replace('speaker_encoder.', '')
                    speaker_encoder_state[new_key] = value
            
            if len(speaker_encoder_state) == 0:
                logging.warning("No speaker encoder weights found in checkpoint!")
                logging.warning(f"Available keys: {list(state_dict.keys())[:10]}...")  # Show first 10 keys
            else:
                logging.info(f"Found {len(speaker_encoder_state)} speaker encoder weights")
                # Load the weights
                self.speaker_encoder.load_state_dict(speaker_encoder_state, strict=True)
                logging.info("Speaker encoder loaded successfully")
        self.freeze_speaker_encoder = freeze_speaker_encoder
        if freeze_speaker_encoder:
            # Freeze speaker encoder parameters
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False
            logging.info("Speaker encoder frozen in flow matching")

        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        print(" decoder_conf['cfm_params']: ", decoder_conf["cfm_params"])
        self.use_contrastive_fm = decoder_conf["cfm_params"]["use_contrastive_fm"]

    def get_speaker_embedding(self, batch, device):
        """Extract speaker embedding from reference mels or use provided embeddings"""
        
        if self.use_speaker_encoder and 'reference_mels' in batch:
            reference_mels = batch['reference_mels'].to(device)
            
            # Handle multiple references
            if reference_mels.dim() == 4:  # [B, N, C, T]
                B, N, C, T = reference_mels.shape
                embeddings = []
                
                for i in range(N):
                    ref_mel = reference_mels[:, i, :, :]  # [B, C, T]
                    if 'reference_mel_masks' in batch:
                        mask = batch['reference_mel_masks'][:, i, :].unsqueeze(1).to(device)
                    else:
                        mask = None
                    # print('ref_mel mask: ', ref_mel.shape, mask.shape)
                    # Apply speaker encoder
                    with torch.set_grad_enabled(not self.freeze_speaker_encoder):
                        emb = self.speaker_encoder(ref_mel, mask)  # [B, spk_embed_dim]
                    embeddings.append(emb)
                
                # Average multiple references
                embedding = torch.stack(embeddings, dim=1).mean(dim=1)  # [B, spk_embed_dim]
                
            else:  # Single reference [B, C, T]
                if 'reference_mel_mask' in batch:
                    mask = batch['reference_mel_mask'].unsqueeze(1).to(device)
                else:
                    mask = None
                
                with torch.set_grad_enabled(not self.freeze_speaker_encoder):
                    embedding = self.speaker_encoder(reference_mels, mask)
            
            # Normalize (already normalized in speaker encoder, but just to be safe)
            embedding = F.normalize(embedding, dim=1)
            
        elif 'embedding' in batch:
            # Use provided embeddings (backward compatibility)
            embedding = batch['embedding'].to(device)
            embedding = F.normalize(embedding, dim=1)
        else:
            # No speaker conditioning
            B = batch['speech_token'].shape[0]
            embedding = torch.zeros(B, self.spk_embed_dim).to(device)
            
        return embedding

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = False  # if random.random() < 0.5 else False
        print("get speaker embedding")
        embedding = self.get_speaker_embedding(batch, device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
        if not self.use_contrastive_fm:
            loss, _ = self.decoder.compute_loss(
                feat.transpose(1, 2).contiguous(),
                mask.unsqueeze(1),
                h.transpose(1, 2).contiguous(),
                embedding,
                cond=conds,
                streaming=streaming,
            )
        else:
            # print("use contrastive fm")
            loss, _ = self.decoder.compute_loss_contrastive(
                feat.transpose(1, 2).contiguous(),
                mask.unsqueeze(1),
                h.transpose(1, 2).contiguous(),
                embedding,
                cond=conds,
                streaming=streaming,
            )
        return {"loss": loss}

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding=None,
        reference_mels=None,
        reference_mel_lengths=None,
        reference_mel_masks=None,
        streaming=False,
        finalize=False,
    ):
        assert token.shape[0] == 1
        
        # Get speaker embedding
        if self.use_speaker_encoder and reference_mels is not None:
            batch = {
                'reference_mels': reference_mels,
                'reference_mel_lengths': reference_mel_lengths,
                'reference_mel_masks': reference_mel_masks
            }
            embedding = self.get_speaker_embedding(batch, token.device)
        elif embedding is not None:
            embedding = F.normalize(embedding, dim=1)
        else:
            embedding = torch.zeros(1, self.spk_embed_dim).to(token.device)
        
        # xvec projection
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = (
            torch.concat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        else:
            token, context = (
                token[:, : -self.pre_lookahead_len],
                token[:, -self.pre_lookahead_len :],
            )
            h, h_lengths = self.encoder(
                token, token_len, context=context, streaming=streaming
            )
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None
