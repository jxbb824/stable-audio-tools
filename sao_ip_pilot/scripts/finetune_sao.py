#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.data.subsets import build_subset_dataset_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config

from common import resolve_repo_path, write_json


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune one SAO full/LOO checkpoint from an explicit include-paths file."
    )
    parser.add_argument("--model-config", default="model_config_freeze_vae.json")
    parser.add_argument("--dataset-config", default="sao_ip_pilot/configs/train_dataset.json")
    parser.add_argument("--val-dataset-config", default="")
    parser.add_argument("--pretrained-ckpt-path", default="dataset/pretrained_model/model.ckpt")
    parser.add_argument("--pretransform-ckpt-path", default="dataset/pretrained_model/vae_model.ckpt")
    parser.add_argument(
        "--remove-pretransform-weight-norm",
        default="none",
        choices=["none", "pre_load", "post_load"],
    )
    parser.add_argument("--output-dir", default="sao_ip_pilot/outputs/models")
    parser.add_argument("--tag", required=True, help="0 for full model; 1..N for LOO models.")
    parser.add_argument("--include-paths-file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--accum-batches", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--strategy", default="")
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--logger", default="none", choices=["none", "wandb", "comet"])
    parser.add_argument("--name", default="sao_ip_pilot")
    parser.add_argument("--use-safetensors", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    path = resolve_repo_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def disable_non_training_callbacks(model_config: dict[str, Any]) -> dict[str, Any]:
    resolved_model_config = deepcopy(model_config)
    training_config = resolved_model_config.setdefault("training", {})
    training_config.setdefault("demo", {})["enabled"] = False
    training_config.setdefault("song_describer_eval", {})["enabled"] = False
    return resolved_model_config


def resolve_logger(logger_name: str, run_name: str, save_dir: str):
    if logger_name == "wandb":
        return pl.loggers.WandbLogger(project=run_name, save_dir=save_dir)
    if logger_name == "comet":
        return pl.loggers.CometLogger(project_name=run_name, save_dir=save_dir)
    return False


def resolve_strategy(args: argparse.Namespace) -> str | pl.strategies.Strategy:
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

            return DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True,
            )
        return args.strategy

    num_devices = max(torch.cuda.device_count(), 1) * args.num_nodes
    return "ddp_find_unused_parameters_true" if num_devices > 1 else "auto"


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()

    output_dir = resolve_repo_path(args.output_dir)
    run_dir = output_dir / str(args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = disable_non_training_callbacks(load_json(args.model_config))
    base_dataset_config = load_json(args.dataset_config)
    include_paths_file = resolve_repo_path(args.include_paths_file)
    if not include_paths_file.exists():
        raise FileNotFoundError(include_paths_file)
    resolved_dataset_config = build_subset_dataset_config(base_dataset_config, str(include_paths_file))
    resolved_dataset_config["drop_last"] = False

    write_json(run_dir / "resolved_model_config.json", model_config)
    write_json(run_dir / "resolved_dataset_config.json", resolved_dataset_config)

    training_seed = int(args.seed)
    if os.environ.get("SLURM_PROCID") is not None:
        training_seed += int(os.environ["SLURM_PROCID"])
    pl.seed_everything(training_seed, workers=True)

    train_dl = create_dataloader_from_config(
        resolved_dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    if args.val_dataset_config:
        val_dataset_config = load_json(args.val_dataset_config)
        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False,
        )
        write_json(run_dir / "resolved_val_dataset_config.json", val_dataset_config)

    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(str(resolve_repo_path(args.pretrained_ckpt_path))))

    if args.remove_pretransform_weight_norm == "pre_load" and getattr(model, "pretransform", None) is not None:
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path and getattr(model, "pretransform", None) is not None:
        model.pretransform.load_state_dict(load_ckpt_state_dict(str(resolve_repo_path(args.pretransform_ckpt_path))))

    if args.remove_pretransform_weight_norm == "post_load" and getattr(model, "pretransform", None) is not None:
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)
    logger = resolve_logger(args.logger, args.name, str(run_dir))

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=resolve_strategy(args),
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ExceptionCallback()],
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        default_root_dir=str(run_dir),
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    trainer.fit(training_wrapper, train_dl, val_dl)
    trainer.strategy.barrier()

    checkpoint_name = "model.safetensors" if args.use_safetensors else "model.ckpt"
    final_checkpoint_path = run_dir / checkpoint_name
    if trainer.is_global_zero:
        training_wrapper.export_model(str(final_checkpoint_path), use_safetensors=args.use_safetensors)
        selected_paths = [
            line.strip()
            for line in include_paths_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        write_json(
            run_dir / "subset_metadata.json",
            {
                "tag": str(args.tag),
                "training_seed": training_seed,
                "model_config": str(resolve_repo_path(args.model_config)),
                "dataset_config": str(resolve_repo_path(args.dataset_config)),
                "resolved_model_config": str(run_dir / "resolved_model_config.json"),
                "resolved_dataset_config": str(run_dir / "resolved_dataset_config.json"),
                "include_paths_file": str(include_paths_file),
                "num_train_paths": len(selected_paths),
                "pretrained_ckpt_path": str(resolve_repo_path(args.pretrained_ckpt_path)),
                "pretransform_ckpt_path": str(resolve_repo_path(args.pretransform_ckpt_path))
                if args.pretransform_ckpt_path
                else None,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "precision": args.precision,
                "accum_batches": args.accum_batches,
                "max_epochs": args.max_epochs,
                "final_checkpoint": str(final_checkpoint_path),
            },
        )
    trainer.strategy.barrier()


if __name__ == "__main__":
    main()
