# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import os
import torch
import json
import re
import datetime
import yaml

import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from loguru import logger
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from cosyvoice.dataset.dataset import Dataset

from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, _LRScheduler

from loguru import logger

class ResumableSequentialLR(_LRScheduler):
    """A resumable version of SequentialLR that properly manages child schedulers"""
    
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            schedulers: List of schedulers to sequentially use
            milestones: List of epoch/step numbers when to switch schedulers
            last_epoch: The index of last epoch/step
        """
        # Validate inputs
        if len(schedulers) != len(milestones) + 1:
            raise ValueError("Expected len(schedulers) == len(milestones) + 1")
        
        self.schedulers = schedulers
        self.milestones = milestones
        self._scheduler_idx = 0
        
        # Initialize parent class (this sets last_epoch and calls step())
        super().__init__(optimizer, last_epoch)
        
    def _get_scheduler_info(self, epoch):
        """Determine which scheduler to use and its relative epoch"""
        scheduler_idx = 0
        relative_epoch = epoch
        
        for i, milestone in enumerate(self.milestones):
            if epoch >= milestone:
                scheduler_idx = i + 1
                if i == 0:
                    relative_epoch = epoch - milestone
                else:
                    relative_epoch = epoch - milestone
            else:
                break
                
        # Calculate relative epoch for the current scheduler
        if scheduler_idx == 0:
            relative_epoch = epoch
        elif scheduler_idx < len(self.milestones):
            if scheduler_idx == 1:
                relative_epoch = epoch - self.milestones[0]
            else:
                relative_epoch = epoch - self.milestones[scheduler_idx - 1]
        
        return scheduler_idx, relative_epoch
    
    def get_lr(self):
        """Get learning rate from the appropriate scheduler"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        # Get current scheduler and its relative epoch
        scheduler_idx, relative_epoch = self._get_scheduler_info(self.last_epoch)
        scheduler = self.schedulers[scheduler_idx]
        
        # Set the scheduler's last_epoch to match relative progress
        scheduler.last_epoch = relative_epoch
        
        # Get LR from the scheduler
        if hasattr(scheduler, '_get_closed_form_lr'):
            return scheduler._get_closed_form_lr()
        else:
            # Temporarily set the flag to avoid warning from child scheduler
            scheduler._get_lr_called_within_step = True
            lrs = scheduler.get_lr()
            scheduler._get_lr_called_within_step = False
            return lrs
    
    def step(self, epoch=None):
        """Step the scheduler"""
        # Step the parent class (updates last_epoch and sets _get_lr_called_within_step)
        super().step(epoch)
        
    def set_step(self, step):
        """Set the current step for resuming training"""
        self.last_epoch = step - 1
        
        # Update child schedulers' state
        scheduler_idx, relative_epoch = self._get_scheduler_info(step - 1)
        
        # Set all previous schedulers to their final state
        for i in range(scheduler_idx):
            if i < len(self.milestones):
                if i == 0:
                    self.schedulers[i].last_epoch = self.milestones[i] - 1
                else:
                    self.schedulers[i].last_epoch = self.milestones[i] - self.milestones[i-1] - 1
        
        # Set current scheduler to its relative position
        self.schedulers[scheduler_idx].last_epoch = relative_epoch
        
        # Update optimizer's learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group['lr'] = lr

def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logger.info(f'training on multiple gpus, this gpu {local_rank}, rank {rank}, world_size {world_size}')
    if args.train_engine == 'torch_ddp':
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, dpo):
    data_pipeline = configs['data_pipeline']
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=False, dpo=dpo, shuffle=True, partition=True)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='train', gan=False, dpo=dpo, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    """Check and modify config"""
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'fp32'
    else:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] *
                                                     configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    """Wrap model to cuda"""
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logger.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(configs, model):
    """Init optimizer and scheduler"""
    lr = configs['train_conf']['optim_conf']['lr']
    logger.info(f"lr base: {lr}")
    if configs['train_conf']['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif configs['train_conf']['optim'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError("unknown optimizer: " + configs['train_conf'])
    
    warm_up_steps = configs['train_conf']['scheduler_conf']['warmup_steps']
    total_iters = configs['train_conf']['total_iters']
    # Create schedulers
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-4,  # Start at nearly 0
        end_factor=1.0,     # End at base learning rate
        total_iters=warm_up_steps    # 5k warmup steps
    )
    
    constant_scheduler = ConstantLR(
        optimizer,
        factor=1.0,  # Keep learning rate constant
        total_iters=total_iters  # Run indefinitely
    )
    
    # Combine schedulers: warmup for 5k steps, then constant
    scheduler = ResumableSequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler],
        milestones=[warm_up_steps]
    )


    return model, optimizer, scheduler



def save_model(model, model_name, info_dict):
    """Save model"""
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))
    

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            torch.save({**model.module.state_dict(), 'epoch': info_dict['epoch'], 'step': info_dict['step']}, save_model_path)
    else:
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=model_name,
                                  client_state=info_dict)
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logger.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))


def cosyvoice_join(group_join, info_dict):
    """Join all ranks"""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=group_join.options._timeout)
            return False
        except RuntimeError as e:
            logger.info("Detected uneven workload distribution: {}\n".format(e) +
                         "Break current worker to manually join all workers, " +
                         "world_size {}, current rank {}, current local_rank {}\n".
                         format(world_size, rank, local_rank))
            return True
    else:
        return False


def batch_forward(model, batch, scaler, info_dict, ref_model=None, dpo_loss=None):
    """ Forward batch and compute loss"""
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        autocast = torch.cuda.amp.autocast(enabled=scaler is not None)
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with autocast:
        info_dict['loss_dict'] = model(batch, device)
        # print('infor_dict loss_dict : ', info_dict['loss_dict'])
        if ref_model is not None and dpo_loss is not None:
            chosen_logps = info_dict['loss_dict']["chosen_logps"]
            rejected_logps = info_dict['loss_dict']["rejected_logps"]
            sft_loss = info_dict['loss_dict']['loss']
            with torch.no_grad():
                ref_loss_dict = ref_model(batch, device)
            reference_chosen_logps = ref_loss_dict["chosen_logps"]
            reference_rejected_logps = ref_loss_dict["rejected_logps"]
            preference_loss, chosen_reward, reject_reward = dpo_loss(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            dpo_acc = (chosen_reward > reject_reward).float().mean()
            info_dict['loss_dict']["loss"] = preference_loss + sft_loss
            info_dict['loss_dict']["sft_loss"] = sft_loss
            info_dict['loss_dict']["dpo_loss"] = preference_loss
            info_dict['loss_dict']["dpo_acc"] = dpo_acc
            info_dict['loss_dict']["chosen_reward"] = chosen_reward.mean()
            info_dict['loss_dict']["reject_reward"] = reject_reward.mean()
    return info_dict


def batch_backward(model, scaler, info_dict):
    """Backward batch"""
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])    
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict, model_type='llm'):
    """Update parameters and learning rate"""

    #Define key components based on model type
    if model_type == 'llm':
        component_patterns = {
            'text_embedding': r'^text_embedding\.',
            'text_encoder': r'^text_encoder\.',
            'text_encoder_affine': r'^text_encoder_affine\.',
            'llm_embedding': r'^llm_embedding\.',
            'llm.model': r'^llm\.model\.',
            'llm_decoder': r'^llm_decoder\.',
            'speech_embedding': r'^speech_embedding\.',
            'spk_embed_affine': r'^spk_embed_affine\.',
        }
    elif model_type == 'flow':
        component_patterns = {
            'input_embedding': r'^input_embedding\.',
            'spk_embed_affine': r'^spk_embed_affine\.',
            'encoder': r'^encoder\.',
            'encoder_proj': r'^encoder_proj\.',
            'decoder.cfm': r'^decoder\..*cfm',
            'decoder.unet': r'^decoder\..*unet',
            'decoder.estimator': r'^decoder\..*estimator',
            'decoder.time_embedding': r'^decoder\..*time_embedding',
            'decoder.conv': r'^decoder\..*conv',
            'decoder.attention': r'^decoder\..*attention',
            'length_regulator': r'^length_regulator\.',
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    key_components = {key: [] for key in component_patterns}
    key_components['other'] = []

    grad_norm = 0.0
    layer_grad_norms = {}

    if (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        # logger.info('start to calculate grad norm')
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Calculate gradient norm for this parameter
                param_grad_norm = param.grad.data.norm(2).item()
                layer_grad_norms[name] = param_grad_norm
                
                # Categorize into key components
                categorized = False
                for component_key in key_components:
                    if component_key != 'other':
                        # Special handling for decoder sub-components in flow models
                        if model_type == 'flow' and component_key.startswith('decoder.'):
                            component_pattern = component_key.replace('decoder.', '')
                            if 'decoder' in name and component_pattern in name:
                                key_components[component_key].append((name, param_grad_norm))
                                categorized = True
                                break
                        elif component_key in name:
                            key_components[component_key].append((name, param_grad_norm))
                            categorized = True
                            break
                if not categorized:
                    key_components['other'].append((name, param_grad_norm))

        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                logger.warning('get infinite grad_norm, check your code/data if it appears frequently')
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logger.warning('get infinite grad_norm, check your code/data if it appears frequently')
        optimizer.zero_grad()
        scheduler.step()
    logger.info(f"lr after step {optimizer.param_groups[0]['lr']}")
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    info_dict["layer_grad_norms"] = layer_grad_norms
    info_dict["key_component_grads"] = key_components
    return info_dict

def log_per_step(experiment, info_dict):
    """Log per step using Comet ML"""
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # Only rank 0 writes to Comet ML to avoid multi-process write
    if experiment is not None and rank == 0:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0):
            # Log metrics to Comet ML
            experiment.log_metric(f'{tag}_epoch', info_dict['epoch'], step=step + 1)
            experiment.log_metric(f'{tag}_lr', info_dict['lr'], step=step + 1)
            experiment.log_metric(f'{tag}_grad_norm', info_dict['grad_norm'], step=step + 1)
            
            # Log all losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                experiment.log_metric(f'{tag}_{k}', v, step=step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = f'{tag} Batch {epoch}/{batch_idx + 1} step {step} '
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_str += f'{name} {value:.6f} '
        if tag == "TRAIN":
            log_str += f'lr {info_dict["lr"]:.15f} grad_norm {info_dict["grad_norm"]:.6f}'
        log_str += f' rank {rank}'
        logger.info(log_str)

def log_per_save(experiment, info_dict):
    """Log per save using Comet ML"""
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    
    # Create loss string for logger
    loss_str = ' '.join([f"{k} {v.item() if isinstance(v, torch.Tensor) else v}" for k, v in loss_dict.items()])
    logger.info(f'Epoch {epoch} Step {step + 1} CV info lr {lr} {rank} {loss_str}')

    if experiment is not None and rank == 0:
        # Log metrics to Comet ML
        experiment.log_metric(f'{tag}_epoch', info_dict['epoch'], step=step + 1)
        experiment.log_metric(f'{tag}_lr', info_dict['lr'], step=step + 1)
        
        # Log all losses
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            experiment.log_metric(f'{tag}_{k}', v, step=step + 1)
        
        # Log additional validation info
        if tag == "CV":
            # Calculate average CV loss for the epoch
            avg_loss = loss_dict.get('loss', 0)
            if isinstance(avg_loss, torch.Tensor):
                avg_loss = avg_loss.item()
            experiment.log_metric('cv_avg_loss_per_epoch', avg_loss, epoch=epoch)
