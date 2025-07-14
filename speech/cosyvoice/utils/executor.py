# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from cosyvoice.utils.train_utils import (batch_backward, batch_forward,
                                         cosyvoice_join, log_per_save,
                                         log_per_step, save_model,
                                         update_parameter_and_lr)

from loguru import logger


class Executor:
    """Executor for training and cross validation"""
    def __init__(
        self,
        gan: bool = False,
        ref_model: torch.nn.Module = None,
        dpo_loss: torch.nn.Module = None,
        use_contrastive_fm: bool = False
    ):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.device = torch.device(f"cuda:{self.rank}")
        self.use_contrastive_fm = use_contrastive_fm

    def train_one_epoc(
        self,
        model,
        optimizer,
        scheduler,
        train_data_loader,
        experiment,
        info_dict,
        scaler,
        model_type
    ):
        """Train one epoch"""

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {self.epoch} TRAIN info lr {lr} rank {self.rank}"
        )
        logger.info(
            f"using accumulate grad, new batch size is {info_dict['accum_grad']} times larger than before"
        )
       
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()

        use_ddp = info_dict["train_engine"] == "torch_ddp"


        for batch_idx, batch_dict in enumerate(train_data_loader):
            info_dict["tag"] = "TRAIN"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx


            if use_ddp and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                context = model.no_sync
            else:
                context = nullcontext


            with context():
                info_dict = batch_forward(
                    model,
                    batch_dict,
                    scaler,
                    info_dict,
                    ref_model=self.ref_model,
                    dpo_loss=self.dpo_loss,
                )

                info_dict = batch_backward(model, scaler, info_dict)

            info_dict = update_parameter_and_lr(
                model, optimizer, scheduler, scaler, info_dict, model_type=model_type
            )
            log_per_step(experiment, info_dict)

            if (
                info_dict.get("save_per_step", -1) > 0
                and (self.step + 1) % info_dict["save_per_step"] == 0
                and (batch_idx + 1) % info_dict["accum_grad"] == 0
            ):
                if dist.is_initialized():
                    dist.barrier()
                model_name = (
                    f"epoch_{self.epoch}_step_{self.step + 1}"
                )
                save_model(model, model_name, info_dict)
                model.train()

            if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.step += 1
        dist.barrier()

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, experiment, info_dict, on_batch_end=True):
        """Cross validation on"""
        logger.info(f"Epoch {self.epoch} Step {self.step + 1} on_batch_end {on_batch_end} CV rank {self.rank}")
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict["turn"] = "generator"
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict["loss_dict"].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict["loss_dict"] = total_loss_dict
        log_per_save(experiment, info_dict)
        model_name = (
            f"epoch_{self.epoch}_whole"
            if on_batch_end
            else f"epoch_{self.epoch}_step_{self.step + 1}"
        )
        save_model(model, model_name, info_dict)
