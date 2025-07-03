import os
import random

import torch
import torch.distributed as dist
from PIL import Image

import utils
from .trainers import register
from trainers.base_trainer import BaseTrainer
from models.ldm.dac.audiotools import AudioSignal
import soundfile as sf
import numpy as np
import torchaudio
import time

from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

@register('audio_ldm_trainer')
class AudioLDMTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def make_model(self):
        super().make_model()
        self.has_optimizer = dict()
        total_params = 0
        for name, m in self.model.named_children():
            params = utils.compute_num_params(m, text=False)
            self.log(f'  .{name} {params}')
            total_params = total_params + params
            # Log to Comet
            if self.experiment:
                self.experiment.log_metric(f"model/{name}_params", params)
        
        if self.experiment:
            self.experiment.log_metric("model/total_params", total_params)

    def make_optimizers(self):
        self.optimizers = dict()
        self.has_optimizer = dict()
        for name, spec in self.config.optimizers.items():
            self.optimizers[name] = utils.make_optimizer(self.model.get_parameters(name), spec)
            self.has_optimizer[name] = True

            # Log optimizer config to Comet
            if self.experiment:
                self.experiment.log_parameters({
                    f"optimizer/{name}/type": spec.get("type", "adam"),
                    f"optimizer/{name}/lr": spec.get("lr", 1e-4),
                    f"optimizer/{name}/weight_decay": spec.get("weight_decay", 0),
                })

    def train_step(self, data, bp=True):
        kwargs = {'has_optimizer': self.has_optimizer}
        
        # Start timing
        step_start_time = time.time()
        # Audio-specific data preparation
        if 'signal' in data:
            # Convert AudioSignal to tensor format expected by model
            audio_data = data['signal'].audio_data  # [batch, channels, samples]
            sample_rate = data['signal'].sample_rate
            
            # Prepare data dict for model
            model_data = {
                'inp': audio_data,
                'gt': audio_data,  # For autoencoder training
                'sample_rate': sample_rate
            }
        else:
            model_data = data
            
        # self.log(f'Audio data shape: {model_data["inp"].shape}')

        # Log batch info to Comet
        if self.experiment and self.iter % 500 == 0:
            self.experiment.log_metric("train/batch_size", model_data["inp"].shape[0], step=self.iter)
            self.experiment.log_metric("train/audio_length_samples", model_data["inp"].shape[-1], step=self.iter)
            self.experiment.log_metric("train/audio_duration_sec", 
                                     model_data["inp"].shape[-1] / model_data.get("sample_rate", 24000), 
                                     step=self.iter)
        

        if self.config.get('autocast_bfloat16', False):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                ret = self.model_ddp(model_data, mode='loss', **kwargs)
        else:
            ret = self.model_ddp(model_data, mode='loss', **kwargs)

        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        
        if bp:
            self.model_ddp.zero_grad(set_to_none=True)
            loss.backward()
            
            # Log gradients to Comet
            if self.experiment and self.iter % 5 == 0:
                self._log_gradients()
            
            for name, o in self.optimizers.items():
                if name != 'disc':
                    o.step()


        if hasattr(self.model, 'update_ema'):
            self.model.update_ema()
        
        # Log training metrics to Comet
        if self.experiment:
            # Log all losses
            for k, v in ret.items():
                if 'loss' in k.lower():
                    self.experiment.log_metric(f"train/{k}", v, step=self.iter)
            
            # Log learning rates
            for name, opt in self.optimizers.items():
                lr = opt.param_groups[0]['lr']
                self.experiment.log_metric(f"train/lr_{name}", lr, step=self.iter)
            
            # Log timing
            step_time = time.time() - step_start_time
            self.experiment.log_metric("train/step_time", step_time, step=self.iter)
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                self.experiment.log_metric("train/gpu_memory_allocated", 
                                         torch.cuda.memory_allocated() / 1e9, 
                                         step=self.iter)
                self.experiment.log_metric("train/gpu_memory_reserved", 
                                         torch.cuda.memory_reserved() / 1e9, 
                                         step=self.iter)
        return ret

    def _log_gradients(self):
        """Log gradient statistics to Comet ML"""
        if not self.experiment:
            return
            
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                # Log aggregate stats by module
                module_name = name.split('.')[0]
                if module_name not in grad_stats:
                    grad_stats[module_name] = {
                        'norm': [],
                        'mean': [],
                        'std': []
                    }
                grad_stats[module_name]['norm'].append(grad_norm)
                grad_stats[module_name]['mean'].append(grad_mean)
                grad_stats[module_name]['std'].append(grad_std)
        
        # Log aggregated stats
        for module, stats in grad_stats.items():
            self.experiment.log_metric(f"gradients/{module}/norm_mean", np.mean(stats['norm']), step=self.iter)
            self.experiment.log_metric(f"gradients/{module}/norm_max", np.max(stats['norm']), step=self.iter)


    def run_training(self):
        config = self.config
        max_iter = config['max_iter']
        epoch_iter = config['epoch_iter']
        assert max_iter % epoch_iter == 0
        max_epoch = max_iter // epoch_iter

        save_iter = config.get('save_iter')
        if save_iter is not None:
            assert save_iter % epoch_iter == 0
            save_epoch = save_iter // epoch_iter
            print('save_epoch', save_epoch)
        else:
            save_epoch = max_epoch + 1

        eval_iter = config.get('eval_iter')
        if eval_iter is not None:
            assert eval_iter % epoch_iter == 0
            eval_epoch = eval_iter // epoch_iter
        else:
            eval_epoch = max_epoch + 1

        vis_iter = config.get('vis_iter')
        if vis_iter is not None:
            assert vis_iter % epoch_iter == 0
            vis_epoch = vis_iter // epoch_iter
        else:
            vis_epoch = max_epoch + 1

        if config.get('ckpt_select_metric') is not None:
            m = config.ckpt_select_metric
            self.ckpt_select_metric = m.name
            self.ckpt_select_type = m.type
            if m.type == 'min':
                self.ckpt_select_v = 1e18
            elif m.type == 'max':
                self.ckpt_select_v = -1e18
        else:
            self.ckpt_select_metric = None
            self.ckpt_select_v = 0

        self.train_loader = self.loaders['train']
        self.train_loader_sampler = self.loader_samplers['train']
        self.train_loader_epoch = 0
        self.train_loader_iter = None

        self.iter = 0

        if self.resume_ckpt is not None:
            for _ in range(self.resume_ckpt['iter']):
                self.iter += 1
                self.at_train_iter_start()
            self.ckpt_select_v = self.resume_ckpt['ckpt_select_v']
            self.train_loader_epoch = self.resume_ckpt['train_loader_epoch']
            self.train_loader_iter = None
            self.resume_ckpt = None
            self.log(f'Resumed iter status.')
        
        self.visualize()

        start_epoch = self.iter // epoch_iter + 1

        for epoch in range(start_epoch, max_epoch + 1):
            self.log_buffer = [f'Epoch {epoch}']

            for sampler in self.loader_samplers.values():
                if sampler is not self.train_loader_sampler:
                    sampler.set_epoch(epoch)

            self.model_ddp.train()

            pbar = range(1, epoch_iter + 1)
            if self.is_master and epoch == start_epoch:
                pbar = tqdm(pbar, desc='train', leave=False)

            t_data = 0
            t_nondata = 0
            t_before_data = time.time()
            
            for _ in pbar:
                self.iter += 1
                self.at_train_iter_start()

                try:
                    if self.train_loader_iter is None:
                        raise StopIteration
                    data = next(self.train_loader_iter)
                except StopIteration:
                    self.train_loader_epoch += 1
                    self.train_loader_sampler.set_epoch(self.train_loader_epoch)
                    self.train_loader_iter = iter(self.train_loader)
                    data = next(self.train_loader_iter)
                
                t_after_data = time.time()
                t_data += t_after_data - t_before_data
                
                for k, v in data.items():
                    data[k] = v.to(self.device) if torch.is_tensor(v) else v
                
                ret = self.train_step(data)
                
                t_before_data = time.time()
                t_nondata += t_before_data - t_after_data

                if self.is_master and epoch == start_epoch:
                    pbar.set_description(desc=f'train: loss={ret["loss"]:.4f}')
                
                # save the model every 1000 iterations
                if self.iter % 2000 == 0:
                    self.save_ckpt(f'ckpt-{self.iter}.pth')
            
            self.save_ckpt('ckpt-last.pth')

            if epoch % save_epoch == 0 and epoch != max_epoch:
                self.save_ckpt(f'ckpt-{self.iter}.pth')

            if epoch % eval_epoch == 0:
                with torch.no_grad():
                    eval_ave_scalars = self.evaluate()
                if self.ckpt_select_metric is not None:
                    v = eval_ave_scalars[self.ckpt_select_metric].item()
                    if ((self.ckpt_select_type == 'min' and v < self.ckpt_select_v) or
                        (self.ckpt_select_type == 'max' and v > self.ckpt_select_v)):
                        self.ckpt_select_v = v
                        self.save_ckpt('ckpt-best.pth')

            if epoch % vis_epoch == 0:
                with torch.no_grad():
                    self.visualize()
    
    def evaluate(self):
        self.model_ddp.eval()

        ave_scalars = dict()
        pbar = self.loaders['val']

        for data in pbar:
            # Prepare audio data for GPU
            if 'signal' in data:
                data['signal'] = data['signal'].to(self.device)
            else:
                for k, v in data.items():
                    data[k] = v.to(self.device) if torch.is_tensor(v) else v
            
            ret = self.train_step(data, bp=False)

            bs = data['signal'].batch_size if 'signal' in data else len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=bs)
        
        self.sync_ave_scalars(ave_scalars)
        
        # Audio-specific evaluation
        if self.config.get('evaluate_ae', False):
            ave_scalars.update(self.evaluate_audio_ae())
        
        if self.config.get('evaluate_zdm', False):
            ema = self.config.get('evaluate_zdm_ema', True)
            ave_scalars.update(self.evaluate_audio_zdm(ema=ema))

        logtext = 'val:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_scalar('val/' + k, v.item())
            
            # Log to Comet
            if self.experiment:
                self.experiment.log_metric(f"val/{k}", v.item(), step=self.iter)
        
        self.log_buffer.append(logtext)
        
        return ave_scalars

    def visualize(self):
        self.model_ddp.eval()

        if self.config.get('evaluate_ae', False):
            self.visualize_audio_ae_random()
        
        if self.config.get('evaluate_zdm', False):
            ema = self.config.get('evaluate_zdm_ema', True)
            self.visualize_audio_zdm_random(ema=ema)

    def evaluate_audio_ae(self):
        """Audio autoencoder evaluation with spectral metrics"""
        max_samples = self.config.get('eval_ae_max_samples', 1000)
        self.loader_samplers['eval_ae'].set_epoch(0)
        
        l1_loss_avg = utils.Averager()
        snr_avg = utils.Averager()
        spectral_convergence_avg = utils.Averager()
        cnt = 0

        # Create cache directories for audio samples
        cache_gen_dir = os.path.join(self.env['save_dir'], 'cache', 'audio_gen')
        cache_gt_dir = os.path.join(self.env['save_dir'], 'cache', 'audio_gt')
        if self.is_master:
            utils.ensure_path(cache_gen_dir, force_replace=True)
            utils.ensure_path(cache_gt_dir, force_replace=True)
        dist.barrier()

        for data in self.loaders['eval_ae']:
            if 'signal' in data:
                data['signal'] = data['signal'].to(self.device)
                signal = data['signal']
            else:
                for k, v in data.items():
                    data[k] = v.to(self.device) if torch.is_tensor(v) else v
                signal = AudioSignal(data['inp'], data.get('sample_rate', 22050))
            
            # Get reconstruction
            pred_audio = self.model(data, mode='pred')
            if isinstance(pred_audio, dict):
                pred_audio = pred_audio.get('audio', pred_audio.get('recons', pred_audio))
            
            recons = AudioSignal(pred_audio, signal.sample_rate)

            # SNR calculation
            signal_power = (signal.audio_data ** 2).mean()
            noise_power = ((recons.audio_data - signal.audio_data) ** 2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            snr_avg.add(snr.item())
            
            # Spectral convergence
            stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=1024,
                hop_length=256,
                power=2
            ).to(self.device)
            
            orig_spec = stft_transform(signal.audio_data)
            recon_spec = stft_transform(recons.audio_data)
            
            spec_diff = torch.norm(orig_spec - recon_spec, p='fro')
            spec_norm = torch.norm(orig_spec, p='fro')
            spectral_convergence = spec_diff / (spec_norm + 1e-8)
            spectral_convergence_avg.add(spectral_convergence.item())
            
            l1_loss = torch.nn.functional.l1_loss(recons.audio_data, signal.audio_data).item()
            l1_loss_avg.add(l1_loss)

            # Save audio samples for potential subjective evaluation
            for i in range(min(signal.batch_size, 5)):  # Save up to 5 per batch
                idx = int(os.environ['RANK']) + cnt * int(os.environ['WORLD_SIZE'])
                if max_samples is None or idx < max_samples:
                    # Save as wav files
                    sf.write(
                        os.path.join(cache_gen_dir, f'{idx}.wav'),
                        recons[i].audio_data.cpu().numpy().T,
                        int(recons[i].sample_rate)
                    )
                    sf.write(
                        os.path.join(cache_gt_dir, f'{idx}.wav'),
                        signal[i].audio_data.cpu().numpy().T,
                        int(signal[i].sample_rate)
                    )
                cnt += 1
    
        dist.barrier()

        # Sync metrics across processes
        for avg_metric in [l1_loss_avg, snr_avg, spectral_convergence_avg]:
            vt = torch.tensor(avg_metric.item(), device=self.device)
            dist.all_reduce(vt, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            avg_metric.v = vt.item() / int(os.environ['WORLD_SIZE'])

        if self.is_master:
            prefix = 'eval_ae'
            ret = {
                f'{prefix}/L1_Loss': l1_loss_avg.item(),
                f'{prefix}/SNR': snr_avg.item(),
                f'{prefix}/Spectral_Convergence': spectral_convergence_avg.item(),
            }
        else:
            ret = {}
        dist.barrier()

        ret = {k: utils.Averager(v) for k, v in ret.items()}
        return ret

    def evaluate_audio_zdm(self, ema):
        """Audio latent diffusion model evaluation"""
        max_samples = self.config.get('eval_zdm_max_samples', 1000)
        self.loader_samplers['eval_zdm'].set_epoch(0)
    
        cnt = 0
        l1_loss_avg = utils.Averager()
        cache_gen_dir = os.path.join(self.env['save_dir'], 'cache', 'audio_gen')
        cache_gt_dir = os.path.join(self.env['save_dir'], 'cache', 'audio_gt')
        if self.is_master:
            utils.ensure_path(cache_gen_dir, force_replace=True)
            utils.ensure_path(cache_gt_dir, force_replace=True)
        dist.barrier()

        for data in self.loaders['eval_zdm']:
            if 'signal' in data:
                data['signal'] = data['signal'].to(self.device)
                gt_signal = data['signal']
            else:
                for k, v in data.items():
                    data[k] = v.to(self.device) if torch.is_tensor(v) else v
                gt_signal = AudioSignal(data['inp'], data.get('sample_rate', 22050))
            
            # Generate samples from latent diffusion model
            net_kwargs = dict()
            uncond_net_kwargs = dict()
            # Add conditioning if available (e.g., for conditional generation)
            
            pred_audio = self.model.generate_samples(
                batch_size=gt_signal.batch_size,
                n_steps=self.model.zdm_n_steps,
                net_kwargs=net_kwargs,
                uncond_net_kwargs=uncond_net_kwargs,
                ema=ema
            )
            
            pred_signal = AudioSignal(pred_audio, gt_signal.sample_rate)

            l1_loss = torch.nn.functional.l1_loss(pred_signal.audio_data, gt_signal.audio_data).item()
            l1_loss_avg.add(l1_loss)

            # Save samples
            for i in range(min(gt_signal.batch_size, 5)):
                idx = int(os.environ['RANK']) + cnt * int(os.environ['WORLD_SIZE'])
                if max_samples is None or idx < max_samples:
                    sf.write(
                        os.path.join(cache_gen_dir, f'{idx}.wav'),
                        pred_signal[i].audio_data.cpu().numpy().T,
                        int(pred_signal[i].sample_rate)
                    )
                    sf.write(
                        os.path.join(cache_gt_dir, f'{idx}.wav'),
                        gt_signal[i].audio_data.cpu().numpy().T,
                        int(gt_signal[i].sample_rate)
                    )
                cnt += 1
        
        dist.barrier()

        # Sync metrics
        for avg_metric in [l1_loss_avg]:
            vt = torch.tensor(avg_metric.item(), device=self.device)
            dist.all_reduce(vt, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            avg_metric.v = vt.item() / int(os.environ['WORLD_SIZE'])

        if self.is_master:
            prefix = 'eval_zdm' + ('_ema' if ema else '')
            ret = {
                f'{prefix}/l1_loss_avg': l1_loss_avg.item(),
            }
        else:
            ret = {}
        dist.barrier()

        ret = {k: utils.Averager(v) for k, v in ret.items()}
        return ret
    
    def visualize_audio_ae_random(self):
        """Save random audio reconstructions for listening"""
        if self.is_master:
            idx_list = list(range(len(self.datasets['eval_ae'])))
            random.shuffle(idx_list)
            n_samples = self.config.get('visualize_ae_random_n_samples', 8)
            
            audio_samples = []
            
            for idx in idx_list[:n_samples]:
                data = self.datasets['eval_ae'][idx]
                
                # Prepare data
                if 'signal' in data:
                    signal = data['signal'].unsqueeze(0).to(self.device)
                    model_data = {
                        'inp': signal.audio_data,
                        'gt': signal.audio_data,
                        'sample_rate': signal.sample_rate
                    }
                else:
                    for k, v in data.items():
                        data[k] = v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v
                    signal = AudioSignal(data['inp'], data.get('sample_rate', 24000))
                    model_data = data
                
                # Get reconstruction
                pred_audio = self.model(model_data, mode='pred')
                if isinstance(pred_audio, dict):
                    pred_audio = pred_audio.get('audio', pred_audio.get('recons', pred_audio))
                
                recons = AudioSignal(pred_audio, signal.sample_rate)
                
                # Save to file and log to Comet
                self.save_audio_sample(signal, f'audio_ae_original_{idx}')
                self.save_audio_sample(recons, f'audio_ae_recons_{idx}')
                
        dist.barrier()
    
    def visualize_audio_zdm_random(self, ema):
        """Save random audio generations from latent diffusion model"""
        if self.is_master:
            n_samples = self.config.get('visualize_zdm_random_n_samples', 8)
            
            for i in range(n_samples):
                # Generate random sample
                net_kwargs = dict()
                uncond_net_kwargs = dict()
                
                # Get a reference from dataset for parameters like sample_rate
                ref_data = self.datasets['eval_ae'][0]
                if 'signal' in ref_data:
                    ref_signal = ref_data['signal']
                    sample_rate = ref_signal.sample_rate
                    batch_size = 1
                else:
                    sample_rate = ref_data.get('sample_rate', 24000)
                    batch_size = 1
                
                pred_audio = self.model.generate_samples(
                    batch_size=batch_size,
                    n_steps=self.model.zdm_n_steps,
                    net_kwargs=net_kwargs,
                    uncond_net_kwargs=uncond_net_kwargs,
                    ema=ema
                )
                
                pred_signal = AudioSignal(pred_audio, sample_rate)
                
                # Save generated audio
                self.save_audio_sample(pred_signal, f'audio_zdm_generated_{i}')
                
        dist.barrier()

    def save_audio_sample(self, audio_signal, name):
        """Save audio sample and log to Comet ML"""
        try:
            # Ensure audio is in correct format
            audio_data = audio_signal.audio_data.cpu()
            
            # Handle different dimensions
            if audio_data.dim() == 3:  # [batch, channels, samples]
                audio_data = audio_data[0]  # Take first sample
            if audio_data.dim() == 2:  # [channels, samples]
                audio_data = audio_data.transpose(0, 1)  # [samples, channels]
            elif audio_data.dim() == 1:  # [samples]
                audio_data = audio_data.unsqueeze(1)  # [samples, 1]
            
            audio_data = audio_data.numpy()
            
            # Normalize if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Save to file
            save_path = os.path.join(self.env['save_dir'], 'audio_samples')
            os.makedirs(save_path, exist_ok=True)
            
            file_path = os.path.join(save_path, f'{name}_step_{self.iter}.wav')
            sf.write(file_path, audio_data, int(audio_signal.sample_rate))
            
            # Log to Comet ML
            if self.experiment:
                self.experiment.log_audio(
                    file_path,
                    metadata={
                        'name': name,
                        'step': self.iter,
                        'sample_rate': int(audio_signal.sample_rate),
                        'duration': len(audio_data) / audio_signal.sample_rate,
                        'channels': audio_data.shape[1] if audio_data.ndim > 1 else 1
                    },
                    step=self.iter
                )
                
                # Also log spectrograms for visualization
                if self.iter % self.config.get('spectrogram_log_freq', 1000) == 0:
                    self._log_spectrogram(audio_signal, name)
            
            self.log(f"Saved audio sample: {file_path}")
            
        except Exception as e:
            self.log(f"Error saving audio sample {name}: {e}")
            if self.experiment:
                self.experiment.log_text(f"Error saving audio {name}: {str(e)}", step=self.iter)
    
    def _log_spectrogram(self, audio_signal, name):
        """Log spectrogram visualization to Comet ML"""
        if not self.experiment:
            return
            
        try:
            
            # Compute spectrogram
            stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=2048,
                hop_length=512,
                power=2
            )
            
            audio_data = audio_signal.audio_data
            if audio_data.dim() == 3:
                audio_data = audio_data[0]
            if audio_data.dim() == 2:
                audio_data = audio_data[0]  # Take first channel
                
            spec = stft_transform(audio_data.cpu())
            spec_db = 10 * torch.log10(spec + 1e-8)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(
                spec_db.numpy(),
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=[0, len(audio_data) / audio_signal.sample_rate, 0, audio_signal.sample_rate / 2]
            )
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'{name} - Spectrogram')
            plt.colorbar(im, ax=ax, label='dB')
            
            # Log to Comet
            self.experiment.log_figure(f"spectrogram/{name}", fig, step=self.iter)
            plt.close(fig)
            
        except Exception as e:
            self.log(f"Error logging spectrogram for {name}: {e}")
    
    
    
    def save_checkpoint(self, tag="latest"):
        """Save checkpoint and log to Comet ML"""
        checkpoint_path = super().save_checkpoint(tag)
        
        if self.experiment and checkpoint_path:
            # Log checkpoint to Comet
            self.experiment.log_model(
                f"checkpoint_{tag}",
                checkpoint_path,
                metadata={
                    "step": self.iter,
                    "tag": tag,
                    "timestamp": datetime.now().isoformat()
                }
            )