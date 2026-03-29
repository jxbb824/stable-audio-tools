#!/usr/bin/env python3

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.data.subsets import (
    build_subset_dataset_config,
    resolve_audio_subset_records,
    select_audio_subset,
    write_subset_records_csv,
)
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one stable-audio fine-tuning run on a deterministic random subset and export the final unwrapped checkpoint."
    )
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--val-dataset-config", type=str, default="")
    parser.add_argument("--pretrained-ckpt-path", type=str, required=True)
    parser.add_argument("--pretransform-ckpt-path", type=str, default="")
    parser.add_argument(
        "--remove-pretransform-weight-norm",
        type=str,
        default="none",
        choices=["none", "pre_load", "post_load"],
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-subdir", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-pool-size", type=int, default=1000)
    parser.add_argument("--train-subset-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--accum-batches", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="")
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--name", type=str, default="stable_audio_groundtruth_finetune")
    parser.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "comet"])
    parser.add_argument("--use-safetensors", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_index_csv(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for value in values:
            writer.writerow([value])


def write_paths_file(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def resolve_logger(logger_name: str, run_name: str, save_dir: str):
    if logger_name == "wandb":
        return pl.loggers.WandbLogger(project=run_name, save_dir=save_dir)
    if logger_name == "comet":
        return pl.loggers.CometLogger(project_name=run_name, save_dir=save_dir)
    return False


def disable_non_training_callbacks(model_config: Dict[str, Any]) -> Dict[str, Any]:
    resolved_model_config = deepcopy(model_config)
    training_config = resolved_model_config.setdefault("training", {})
    training_config.setdefault("demo", {})["enabled"] = False
    training_config.setdefault("song_describer_eval", {})["enabled"] = False
    return resolved_model_config


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


def build_subset_artifacts(
    args: argparse.Namespace,
    dataset_config: Dict[str, Any],
    run_dir: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    all_records = resolve_audio_subset_records(dataset_config)
    candidate_records, selected_pool_indices, selected_records = select_audio_subset(
        all_records,
        candidate_pool_size=args.candidate_pool_size,
        train_subset_size=args.train_subset_size,
        seed=args.seed,
    )

    candidate_manifest_path = run_dir / "candidate_pool.csv"
    train_manifest_path = run_dir / "train_manifest.csv"
    train_index_path = run_dir / "train_index.csv"
    train_index_global_path = run_dir / "train_index_global.csv"
    selected_paths_path = run_dir / "selected_paths.txt"

    write_subset_records_csv(candidate_manifest_path, candidate_records)
    write_subset_records_csv(train_manifest_path, selected_records, pool_indices=selected_pool_indices)
    write_index_csv(train_index_path, selected_pool_indices)
    write_index_csv(train_index_global_path, [record.global_index for record in selected_records])
    write_paths_file(selected_paths_path, [record.path for record in selected_records])

    resolved_dataset_config = build_subset_dataset_config(dataset_config, str(selected_paths_path))
    return (
        resolved_dataset_config,
        {
            "candidate_manifest_path": str(candidate_manifest_path),
            "train_manifest_path": str(train_manifest_path),
            "train_index_path": str(train_index_path),
            "train_index_global_path": str(train_index_global_path),
            "selected_paths_path": str(selected_paths_path),
        },
        {
            "candidate_records": candidate_records,
            "selected_pool_indices": selected_pool_indices,
            "selected_records": selected_records,
        },
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    run_name = args.output_subdir if args.output_subdir else str(args.run_id)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_model_config = load_json(args.model_config)
    base_dataset_config = load_json(args.dataset_config)
    resolved_model_config = disable_non_training_callbacks(base_model_config)
    resolved_dataset_config, artifact_paths, subset_state = build_subset_artifacts(
        args=args,
        dataset_config=base_dataset_config,
        run_dir=run_dir,
    )

    resolved_model_config_path = run_dir / "resolved_model_config.json"
    resolved_dataset_config_path = run_dir / "resolved_dataset_config.json"
    save_json(resolved_model_config_path, resolved_model_config)
    save_json(resolved_dataset_config_path, resolved_dataset_config)

    val_dataset_config = None
    if args.val_dataset_config:
        val_dataset_config = load_json(args.val_dataset_config)
        save_json(run_dir / "resolved_val_dataset_config.json", val_dataset_config)

    selection_seed = args.seed
    training_seed = args.seed
    if os.environ.get("SLURM_PROCID") is not None:
        training_seed += int(os.environ["SLURM_PROCID"])
    pl.seed_everything(training_seed, workers=True)

    train_dl = create_dataloader_from_config(
        resolved_dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=resolved_model_config["sample_rate"],
        sample_size=resolved_model_config["sample_size"],
        audio_channels=resolved_model_config.get("audio_channels", 2),
    )

    val_dl = None
    if val_dataset_config is not None:
        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=resolved_model_config["sample_rate"],
            sample_size=resolved_model_config["sample_size"],
            audio_channels=resolved_model_config.get("audio_channels", 2),
            shuffle=False,
        )

    model = create_model_from_config(resolved_model_config)
    copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load" and getattr(model, "pretransform", None) is not None:
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path and getattr(model, "pretransform", None) is not None:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    if args.remove_pretransform_weight_norm == "post_load" and getattr(model, "pretransform", None) is not None:
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(resolved_model_config, model)
    logger = resolve_logger(args.logger, args.name, str(run_dir))

    run_hparams = {
        "run_id": args.run_id,
        "output_subdir": args.output_subdir or None,
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "candidate_pool_size": args.candidate_pool_size,
        "train_subset_size": args.train_subset_size,
        "max_epochs": args.max_epochs,
    }
    if hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(run_hparams)

    resolved_strategy = resolve_strategy(args)
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=resolved_strategy,
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

        metadata = {
            "run_id": args.run_id,
            "output_subdir": args.output_subdir or None,
            "selection_seed": selection_seed,
            "training_seed": training_seed,
            "base_model_config": str(Path(args.model_config).resolve()),
            "base_dataset_config": str(Path(args.dataset_config).resolve()),
            "resolved_model_config": str(resolved_model_config_path),
            "resolved_dataset_config": str(resolved_dataset_config_path),
            "val_dataset_config": str(Path(args.val_dataset_config).resolve()) if args.val_dataset_config else None,
            "pretrained_ckpt_path": str(Path(args.pretrained_ckpt_path).resolve()),
            "pretransform_ckpt_path": str(Path(args.pretransform_ckpt_path).resolve()) if args.pretransform_ckpt_path else None,
            "candidate_pool_size": args.candidate_pool_size,
            "train_subset_size": args.train_subset_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "precision": args.precision,
            "accum_batches": args.accum_batches,
            "num_nodes": args.num_nodes,
            "strategy": args.strategy or resolved_strategy,
            "gradient_clip_val": args.gradient_clip_val,
            "max_epochs": args.max_epochs,
            "logger": args.logger,
            "final_checkpoint": str(final_checkpoint_path),
            "artifacts": artifact_paths,
            "selected_pool_indices": subset_state["selected_pool_indices"],
            "selected_global_indices": [record.global_index for record in subset_state["selected_records"]],
            "selected_track_ids": [record.track_id for record in subset_state["selected_records"]],
        }
        save_json(run_dir / "subset_metadata.json", metadata)

    trainer.strategy.barrier()


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
