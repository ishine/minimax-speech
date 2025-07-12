# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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

from __future__ import print_function

import argparse
import datetime
import os
from copy import deepcopy

import deepspeed
import torch
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from loguru import logger
from torch.distributed.elastic.multiprocessing.errors import record

from comet_ml import Experiment
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.train_utils import (check_modify_and_save_config,
                                         init_dataset_and_dataloader,
                                         init_distributed,
                                         init_optimizer_and_scheduler,
                                         init_summarywriter, save_model)


def get_args():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument(
        "--train_engine",
        default="torch_ddp",
        choices=["torch_ddp", "deepspeed"],
        help="Engine for paralleled training",
    )
    parser.add_argument("--model", required=True, help="model which will be trained")
    parser.add_argument("--ref_model", required=False, help="ref model used in dpo")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--train_data", required=True, help="train data file")
    parser.add_argument("--cv_data", required=True, help="cv data file")
    parser.add_argument(
        "--qwen_pretrain_path", required=False, help="qwen pretrain path"
    )
    parser.add_argument("--checkpoint", help="checkpoint model")
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument(
        "--tensorboard_dir", default="tensorboard", help="tensorboard log dir"
    )
    parser.add_argument(
        "--ddp.dist_backend",
        dest="dist_backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="distributed backend",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="num of subprocess workers for reading",
    )
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Use pinned memory buffers used for reading",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--dpo",
        action="store_true",
        default=False,
        help="Use Direct Preference Optimization",
    )
    parser.add_argument(
        "--deepspeed.save_states",
        dest="save_states",
        default="model_only",
        choices=["model_only", "model+optimizer"],
        help="save model/optimizer states",
    )
    parser.add_argument(
        "--timeout",
        default=60,
        type=int,
        help="timeout (in seconds) of cosyvoice_join.",
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def init_comet_experiment(args, configs):
    """Initialize Comet ML experiment"""
    rank = int(os.environ.get('RANK', 0))
    
    # Only create experiment on rank 0 to avoid duplicates
    if rank == 0 and not args.comet_disabled:
        # Set up Comet ML experiment
        experiment = Experiment(
            api_key=args.comet_api_key,
            project_name=args.comet_project,
            workspace=args.comet_workspace,
            experiment_name=args.comet_experiment_name,
            disabled=args.comet_disabled,
            offline=args.comet_offline,
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=False,
        )
        
        # Log hyperparameters
        experiment.log_parameters(configs["train_conf"])
        experiment.log_parameter("model_type", args.model)
        experiment.log_parameter("train_data", args.train_data)
        experiment.log_parameter("cv_data", args.cv_data)
        experiment.log_parameter("use_amp", args.use_amp)
        experiment.log_parameter("dpo", args.dpo)
        experiment.log_parameter("num_workers", args.num_workers)
        experiment.log_parameter("prefetch", args.prefetch)
        
        # Log model architecture if available
        if args.model in configs:
            model_config = configs[args.model].__dict__ if hasattr(configs[args.model], '__dict__') else {}
            experiment.log_parameters(model_config, prefix=f"{args.model}/")
        
        # Add tags
        experiment.add_tag(args.model)
        if args.dpo:
            experiment.add_tag("dpo")
        if args.use_amp:
            experiment.add_tag("amp")
            
        logger.info(f"Comet ML experiment initialized: {experiment.get_name()}")
        return experiment
    else:
        return None

@record
def main():
    args = get_args()

    override_dict = {
        k: None for k in ["llm", "flow", "hift", "hifigan"] if k != args.model
    }
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    **override_dict,
                    "qwen_pretrain_path": args.qwen_pretrain_path,
                },
            )
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        with open(args.config, "r", encoding="utf-8") as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)

    configs["train_conf"].update(vars(args))

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logger.info(f'training on multiple gpus, this gpu {local_rank}, rank {rank}, world_size {world_size}')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.dist_backend)

    # Get dataset & dataloader
    train_dataset, _, train_data_loader, cv_data_loader = init_dataset_and_dataloader(
        args, configs, args.dpo
    )

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    experiment = init_comet_experiment(args, configs)


    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo

    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            if "step" in state_dict:
                start_step = state_dict["step"]
            if "epoch" in state_dict:
                start_epoch = state_dict["epoch"]
            # Log checkpoint info to Comet
            if experiment:
                experiment.log_parameter("checkpoint", args.checkpoint)
                experiment.log_parameter("start_step", start_step)
                experiment.log_parameter("start_epoch", start_epoch)
        else:
            logger.warning(f"checkpoint {args.checkpoint} do not exsist!")

    # Dispatch model from cpu to gpu
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True
    )

    # Get optimizer & scheduler
    model, optimizer, scheduler = (
        init_optimizer_and_scheduler(configs, model)
    )
    scheduler.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs["train_conf"])
    info_dict["step"] = start_step
    info_dict["epoch"] = start_epoch
    save_model(model, "init", info_dict)

    # Log model save to Comet
    if experiment:
        experiment.log_model(
            name=f"{args.model}_init",
            file_or_folder=os.path.join(args.model_dir, "init.pt"),
            metadata=info_dict
        )

    # DPO related
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location="cpu")
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        ref_model = ref_model.cuda()
        ref_model = torch.nn.parallel.DistributedDataParallel(
            ref_model, find_unused_parameters=True
        )
        if experiment:
            experiment.log_parameter("ref_model", args.ref_model)
            experiment.log_parameter("dpo_beta", 0.01)
            experiment.log_parameter("dpo_label_smoothing", 0.0)
            experiment.log_parameter("dpo_ipo", False)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=False, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.amp.GradScaler() if args.use_amp else None
    logger.info(f"start step {start_step} start epoch {start_epoch}")

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict["max_epoch"]):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(
            backend="nccl", timeout=datetime.timedelta(seconds=args.timeout)
        )
        
        executor.train_one_epoc(
            model,
            optimizer,
            scheduler,
            train_data_loader,
            cv_data_loader,
            experiment,
            info_dict,
            scaler,
            group_join,
            model_type=args.model
        )
        dist.destroy_process_group(group_join)
    if experiment:
        experiment.end()

if __name__ == "__main__":
    main()
