import argparse
import json
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import toml
import warnings
import datetime

warnings.filterwarnings("ignore")

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from time import gmtime, strftime
from utils.utils import *
from utils.metric import Metric
from dataloader.dataset import DefaultCollate
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)

from audiomentations import (
    AddBackgroundNoise,
    PolarityInversion,
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
)
import numpy as np
from importlib.machinery import SourceFileLoader
from transformers.file_utils import cached_path, hf_bucket_url


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "4444"

    # initialize the process group
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600 * 5),
    )


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config, resume, preload, noise_path, pretrained_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["meta"]["device_ids"]
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    setup(rank, world_size)

    epochs = config["meta"]["epochs"]
    gradient_accumulation_steps = config["meta"]["gradient_accumulation_steps"]
    use_amp = config["meta"]["use_amp"]
    max_clip_grad_norm = config["meta"]["max_clip_grad_norm"]
    save_dir = os.path.join(
        config["meta"]["save_dir"], config["meta"]["name"] + "/checkpoints"
    )
    log_dir = os.path.join(
        config["meta"]["save_dir"], config["meta"]["name"] + "/log_dir"
    )

    if rank == 0:
        # Creatr dirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Store config file
        config_name = (
            strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "_") + ".toml"
        )
        with open(
            os.path.join(
                config["meta"]["save_dir"], config["meta"]["name"] + "/" + config_name
            ),
            "w+",
        ) as f:
            toml.dump(config, f)
            f.close()

    # This should be needed to be reproducible https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    config["meta"]["seed"] += rank
    set_seed(config["meta"]["seed"])
    config["val_dataset"]["args"]["sr"] = config["meta"]["sr"]
    config["train_dataset"]["args"]["sr"] = config["meta"]["sr"]

    config["train_dataset"]["args"]["rank"] = rank
    config["val_dataset"]["args"]["rank"] = rank

    config["train_dataset"]["args"]["dist"] = dist
    config["val_dataset"]["args"]["dist"] = dist

    config["train_dataset"]["args"]["special_tokens"] = config["special_tokens"]
    config["val_dataset"]["args"]["special_tokens"] = config["special_tokens"]

    train_base_ds = initialize_module(
        config["train_dataset"]["path"], args=config["train_dataset"]["args"]
    )

    dist.barrier()
    # Create processor

    processor = Wav2Vec2Processor.from_pretrained(pretrained_path)

    noise_transform = AddBackgroundNoise(
        sounds_path=noise_path,
        noise_rms="absolute",
        min_absolute_rms_in_db=-45.0,
        max_absolute_rms_in_db=-15.0,
        noise_transform=PolarityInversion(),
        p=0.5,
    )

    aug_transform = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(p=0.5),
        ]
    )

    default_collate = DefaultCollate(
        processor, config["meta"]["sr"], noise_transform, aug_transform
    )

    # Create train dataloader
    train_ds = train_base_ds.get_data()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        **config["train_dataset"]["sampler"]
    )
    train_dl = DataLoader(
        dataset=train_ds,
        **config["train_dataset"]["dataloader"],
        sampler=train_sampler,
        collate_fn=default_collate
    )

    # Create val dataloader
    val_base_ds = initialize_module(
        config["val_dataset"]["path"], args=config["val_dataset"]["args"]
    )
    val_ds = val_base_ds.get_data()
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, **config["val_dataset"]["sampler"]
    )
    val_dl = DataLoader(
        dataset=val_ds,
        **config["val_dataset"]["dataloader"],
        sampler=val_sampler,
        collate_fn=default_collate
    )

    # Load pretrained model
    model = (
        SourceFileLoader(
            "model",
            cached_path(hf_bucket_url(pretrained_path, filename="model_handling.py")),
        )
        .load_module()
        .Wav2Vec2ForCTC.from_pretrained(
            pretrained_path,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            # gradient_checkpointing=False
        )
    )

    # freeze the wav2vec feature encoder, if you have small dataset, this helps a lot
    model.freeze_feature_encoder()
    # DDP for multi-processing
    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    # Set up metric, scheduler, optmizer
    compute_metric = Metric(processor)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=config["optimizer"]["lr"]
    )
    steps_per_epoch = (len(train_dl) // gradient_accumulation_steps) + (
        len(train_dl) % gradient_accumulation_steps != 0
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["scheduler"]["max_lr"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    if rank == 0:
        print("Number of training utterances: ", len(train_ds))
        print("Number of validation utterances: ", len(val_ds))

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist=dist,
        rank=rank,
        n_gpus=world_size,
        config=config,
        resume=resume,
        preload=preload,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        model=model,
        compute_metric=compute_metric,
        processor=processor,
        train_dl=train_dl,
        val_dl=val_dl,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        log_dir=log_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=use_amp,
        max_clip_grad_norm=max_clip_grad_norm,
    )
    trainer.train()

    cleanup()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="ASR TRAIN ARGS")
    args.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-p", "--preload", default=None, type=str, help="Path to pretrained Model"
    )
    args.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Path of huggingface pretrained model",
    )
    args.add_argument("--noise_path", default=None, type=str, help="Path to noise data")

    args = args.parse_args()
    config = toml.load(args.config)
    n_gpus = len(config["meta"]["device_ids"].split(","))

    mp.spawn(
        main,
        args=(
            n_gpus,
            config,
            args.resume,
            args.preload,
            args.noise_path,
            args.pretrained_path,
        ),
        nprocs=n_gpus,
        join=True,
    )
