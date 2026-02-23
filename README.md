# Jamendo Run Guide
This section documents the full workflow used in this repo for MTG-Jamendo style training, deduplication, attribution, inference, and evaluation.

Everything below is written for Jamendo data layout first. A "using your own dataset" checklist is included near the end.

## 1) Environment Setup
There is no single locked requirements file in this repo. The practical setup is:

```bash
# from repo root
conda create -n sat-jamendo python=3.10 -y
conda activate sat-jamendo

# pick the torch build that matches your CUDA runtime
pip install --upgrade pip
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# install this repo
pip install -e .

# LoRA submodule/package (needed only for LoRA training)
git submodule update --init --recursive
pip install -e third_party/loraw
```

Install the dependencies in one command:

```bash
pip install prefigure descript-audio-codec laion-clap transformers k-diffusion v-diffusion-pytorch vector-quantize-pytorch alias-free-torch ema-pytorch traker fast_jl bitsandbytes wandb -e third_party/loraw
```

## 2) Jamendo Data Layout
Expected layout in this repo:

```text
dataset/
  pretrained_model/
    model.ckpt
    vae_model.ckpt
  small-5000/
    audio/
      00/*.mp3
      01/*.mp3
      ...
    raw_30s.tsv
    raw.meta.tsv
  small-700/
    audio/
      00/*.2min.mp3
      01/*.2min.mp3
      ...
    raw.tsv
    audio_metadata.tsv
    song_describer.csv
    song_describer_clap_embeddings.npz
```

What each folder is used for:

- `dataset/small-5000`: training set
- `dataset/small-700`: song_describer/CLAP evaluation set
- `dataset/pretrained_model`: base checkpoint + VAE checkpoint used for finetuning

Important:

- `download_mtg_jamendo.py` downloads audio only.
- `raw_30s.tsv` and `raw.meta.tsv` are used by `custom_md_fma.py` to create text prompts.

## 3) Download or Prepare Jamendo Audio
To download around 5000 tracks into training layout:

```bash
python3 download_mtg_jamendo.py \
  --target 5000 \
  --audio-dir dataset/small-5000/audio
```

After that, make sure metadata files exist:

- `dataset/small-5000/raw_30s.tsv`
- `dataset/small-5000/raw.meta.tsv`

## 4) Deduplicate the Training Set
Run dedup:

```bash
python3 scripts/dac_hist_dedup.py \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_training_custom.json \
  --output-dir outputs/dac_dedup \
  --device cuda \
  --similarity-device cuda \
  --similarity-threshold 0.95 \
  --histogram-mode per_codebook \
  --max-audio-seconds 120
```

Main outputs:

- `outputs/dac_dedup/exclude_paths.txt`
- `outputs/dac_dedup/similar_pairs.jsonl`
- `outputs/dac_dedup/summary.json`

`stable_audio_tools/configs/dataset_configs/local_training_custom.json` already points `exclude_paths_file` to `outputs/dac_dedup/exclude_paths.txt`.

## 5) Precompute Song-Describer CLAP Embeddings
`model_config*.json` in this repo enables `training.song_describer_eval` by default, so precomputed CLAP embeddings should exist before training.

```bash
python3 scripts/precompute_song_describer_clap_embeddings.py \
  --csv-path dataset/small-700/song_describer.csv \
  --audio-dir dataset/small-700/audio \
  --output-path dataset/small-700/song_describer_clap_embeddings.npz \
  --prompt-source training_style \
  --prompt-seed 0 \
  --device cuda
```

If you want to skip this evaluation during training, set `training.song_describer_eval.enabled` to `false` in your model config.

## 6) Training Modes
Common dataset config used below:

- `stable_audio_tools/configs/dataset_configs/local_training_custom.json`

### 6.1 Full finetune (standard)
```bash
python3 train.py \
  --model-config model_config.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_training_custom.json \
  --pretrained-ckpt-path dataset/pretrained_model/model.ckpt \
  --pretransform-ckpt-path dataset/pretrained_model/vae_model.ckpt \
  --batch-size 8 \
  --num-workers 8 \
  --precision "16-mixed" \
  --name "stable_audio_open_finetune" \
  --save-dir outputs
```

### 6.2 Finetune with frozen VAE path
```bash
python3 train.py \
  --model-config model_config_freeze_vae.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_training_custom.json \
  --pretrained-ckpt-path dataset/pretrained_model/model.ckpt \
  --pretransform-ckpt-path dataset/pretrained_model/vae_model.ckpt \
  --batch-size 8 \
  --num-workers 8 \
  --precision "16-mixed" \
  --name "stable_audio_open_finetune_freeze_vae" \
  --save-dir outputs
```

### 6.3 LoRA finetune
Use `train_lora.py`:

```bash
python3 train_lora.py \
  --model-config model_config_lora.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_training_custom.json \
  --pretrained-ckpt-path dataset/pretrained_model/model.ckpt \
  --pretransform-ckpt-path dataset/pretrained_model/vae_model.ckpt \
  --batch-size 32 \
  --num-workers 8 \
  --precision "16-mixed" \
  --name "stable_audio_lora_jamendo" \
  --save-dir outputs
```

## 7) Unwrap Checkpoint Before Inference / External Eval
Training checkpoints are Lightning wrappers. Export an unwrapped model checkpoint:

```bash
python3 unwrap_model.py \
  --model-config model_config.json \
  --ckpt-path outputs/stable_audio_open_finetune/<run_id>/checkpoints/<checkpoint>.ckpt \
  --name model_unwrap
```

This writes `model_unwrap.ckpt` in repo root. Use this for inference and eval scripts.

## 8) Minimal Inference Script

```python
import json
from pathlib import Path

import torch
import torchaudio
from einops import rearrange

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

CONFIG_PATH = "model_config.json"
UNWRAPPED_CKPT = "model_unwrap.ckpt"
PROMPTS = [
    "dirty, rock, breakbeat, metal",
    "reggae, rock",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path("outputs/generated")
out_dir.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_config = json.load(f)

model = create_model_from_config(model_config)
model.load_state_dict(load_ckpt_state_dict(UNWRAPPED_CKPT), strict=False)
model = model.to(device).eval()

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
seconds_total = sample_size / sample_rate

for i, prompt in enumerate(PROMPTS):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0.0,
        "seconds_total": seconds_total,
    }]
    audio = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        batch_size=1,
        sample_size=sample_size,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.3,
        sigma_max=500.0,
        device=device,
        seed=42 + i,
    )

    audio = rearrange(audio, "b c t -> c (b t)").to(torch.float32)
    peak = audio.abs().max().clamp(min=1e-8)
    audio = (audio / peak).clamp(-1, 1)
    audio_i16 = (audio * 32767.0).to(torch.int16).cpu()

    torchaudio.save(str(out_dir / f"output_finetuned_{i}.wav"), audio_i16, sample_rate)
```

## 9) Data Attribution (D-TRAK) Workflow
`scripts/run_dtrak_attribution.sh` expects query wavs in `outputs/generated` and by default uses:

- train set: `stable_audio_tools/configs/dataset_configs/local_training_custom.json`
- query set: `stable_audio_tools/configs/dataset_configs/dtrak_generated_queries.json`

Run:

```bash
bash scripts/run_dtrak_attribution.sh
```

Or with Slurm:

```bash
sbatch scripts/run_dtrak_attribution.slurm
```

Note on prompt mapping for queries:

- The current default query metadata mapping is broken for real use (it was hardcoded for an earlier smoketest).
- You should treat `custom_md_generated_from_test.py` as a placeholder and replace it with your own query-to-prompt mapping.
- Update `stable_audio_tools/configs/dataset_configs/dtrak_generated_queries.json` to point to your own metadata module.

## 10) Standalone Evaluation (CLAP-FAD + CLAP Alignment)
After you have an unwrapped checkpoint:

```bash
python3 scripts/eval_song_describer_checkpoint_clap.py \
  --model-config model_config.json \
  --ckpt-path model_unwrap.ckpt \
  --precomputed-embeddings-path dataset/small-700/song_describer_clap_embeddings.npz \
  --steps 100 \
  --cfg-scale 7 \
  --gen-batch-size 4 \
  --clap-batch-size 8 \
  --device cuda
```

Or run:

```bash
sbatch scripts/run_eval_song_describer_checkpoint_clap.slurm
```

## 11) If You Switch to a New Dataset (Non-Jamendo)
You can reuse:

- `train.py`, `train_lora.py`, `unwrap_model.py`, `scripts/dac_hist_dedup.py`
- D-TRAK scripts (`scripts/dtrak_extract_features.py`, `scripts/dtrak_score.py`)
- inference API (`stable_audio_tools.inference.generation.generate_diffusion_cond`)

You must update:

- Dataset paths in your dataset config JSON.
- `custom_metadata_module` so that `get_custom_metadata(info, audio)` returns at least a meaningful `prompt`.
- If you need query attribution, query dataset config and query metadata mapping.
- Song-describer evaluation assets (`song_describer.csv`, ref audio, precomputed CLAP embeddings), or disable `training.song_describer_eval`.

---

# stable-audio-tools
Training and inference code for audio generation models

# Install

The library can be installed from PyPI with:
```bash
$ pip install stable-audio-tools
```

To run the training scripts or inference code, you'll want to clone this repository, navigate to the root, and run:
```bash
$ pip install .
```

# Requirements
Requires PyTorch 2.5 or later for Flash Attention and Flex Attention support

Development for the repo is done in Python 3.10

# Interface

A basic Gradio interface is provided to test out trained models. 

For example, to create an interface for the [`stable-audio-open-1.0`](https://huggingface.co/stabilityai/stable-audio-open-1.0) model, once you've accepted the terms for the model on Hugging Face, you can run:
```bash
$ python3 ./run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0
```

The `run_gradio.py` script accepts the following command line arguments:

- `--pretrained-name`
  - Hugging Face repository name for a Stable Audio Tools model
  - Will prioritize `model.safetensors` over `model.ckpt` in the repo
  - Optional, used in place of `model-config` and `ckpt-path` when using pre-trained model checkpoints on Hugging Face
- `--model-config`
  - Path to the model config file for a local model
- `--ckpt-path`
  - Path to unwrapped model checkpoint file for a local model
- `--pretransform-ckpt-path` 
  - Path to an unwrapped pretransform checkpoint, replaces the pretransform in the model, useful for testing out fine-tuned decoders
  - Optional
- `--share`
  - If true, a publicly shareable link will be created for the Gradio demo
  - Optional
- `--username` and `--password`
  - Used together to set a login for the Gradio demo
  - Optional
- `--model-half`
  - If true, the model weights to half-precision
  - Optional

# Training

## Prerequisites
Before starting your training run, you'll need a model config file, as well as a dataset config file. For more information about those, refer to the Configurations section below

The training code also requires a Weights & Biases account to log the training outputs and demos. Create an account and log in with:
```bash
$ wandb login
```

## Start training
To start a training run, run the `train.py` script in the repo root with:
```bash
$ python3 ./train.py --dataset-config /path/to/dataset/config --model-config /path/to/model/config --name harmonai_train
```

The `--name` parameter will set the project name for your Weights and Biases run.

## Training wrappers and model unwrapping
`stable-audio-tools` uses PyTorch Lightning to facilitate multi-GPU and multi-node training. 

When a model is being trained, it is wrapped in a "training wrapper", which is a `pl.LightningModule` that contains all of the relevant objects needed only for training. That includes things like discriminators for autoencoders, EMA copies of models, and all of the optimizer states.

The checkpoint files created during training include this training wrapper, which greatly increases the size of the checkpoint file.

`unwrap_model.py` in the repo root will take in a wrapped model checkpoint and save a new checkpoint file including only the model itself.

That can be run with from the repo root with:
```bash
$ python3 ./unwrap_model.py --model-config /path/to/model/config --ckpt-path /path/to/wrapped/ckpt --name model_unwrap
```

Unwrapped model checkpoints are required for:
  - Inference scripts
  - Using a model as a pretransform for another model (e.g. using an autoencoder model for latent diffusion)
  - Fine-tuning a pre-trained model with a modified configuration (i.e. partial initialization)

## Fine-tuning
Fine-tuning a model involves continuning a training run from a pre-trained checkpoint. 

To continue a training run from a wrapped model checkpoint, you can pass in the checkpoint path to `train.py` with the `--ckpt-path` flag.

To start a fresh training run using a pre-trained unwrapped model, you can pass in the unwrapped checkpoint to `train.py` with the `--pretrained-ckpt-path` flag.

## Additional training flags

Additional optional flags for `train.py` include:
- `--config-file`
  - The path to the defaults.ini file in the repo root, required if running `train.py` from a directory other than the repo root
- `--pretransform-ckpt-path`
  - Used in various model types such as latent diffusion models to load a pre-trained autoencoder. Requires an unwrapped model checkpoint.
- `--save-dir`
  - The directory in which to save the model checkpoints
- `--checkpoint-every`
  - The number of steps between saved checkpoints.
  - *Default*: 10000
- `--batch-size`
  - Number of samples per-GPU during training. Should be set as large as your GPU VRAM will allow.
  - *Default*: 8
- `--num-gpus`
  - Number of GPUs per-node to use for training
  - *Default*: 1
- `--num-nodes`
  - Number of GPU nodes being used for training
  - *Default*: 1
- `--accum-batches`
  - Enables and sets the number of batches for gradient batch accumulation. Useful for increasing effective batch size when training on smaller GPUs.
- `--strategy`
  - Multi-GPU strategy for distributed training. Setting to `deepspeed` will enable DeepSpeed ZeRO Stage 2.
  - *Default*: `ddp` if `--num_gpus` > 1, else None
- `--precision`
  - floating-point precision to use during training
  - *Default*: 16
- `--num-workers`
  - Number of CPU workers used by the data loader
- `--seed`
  - RNG seed for PyTorch, helps with deterministic training

# Configurations
Training and inference code for `stable-audio-tools` is based around JSON configuration files that define model hyperparameters, training settings, and information about your training dataset.

## Model config
The model config file defines all of the information needed to load a model for training or inference. It also contains the training configuration needed to fine-tune a model or train from scratch.

The following properties are defined in the top level of the model configuration:

- `model_type`
  - The type of model being defined, currently limited to one of `"autoencoder", "diffusion_uncond", "diffusion_cond", "diffusion_cond_inpaint", "diffusion_autoencoder", "lm"`.
- `sample_size`
  - The length of the audio provided to the model during training, in samples. For diffusion models, this is also the raw audio sample length used for inference.
- `sample_rate`
  - The sample rate of the audio provided to the model during training, and generated during inference, in Hz.
- `audio_channels`
  - The number of channels of audio provided to the model during training, and generated during inference. Defaults to 2. Set to 1 for mono.
- `model`
  - The specific configuration for the model being defined, varies based on `model_type`
- `training`
  - The training configuration for the model, varies based on `model_type`. Provides parameters for training as well as demos.

## Dataset config
`stable-audio-tools` currently supports two kinds of data sources: local directories of audio files, and WebDataset datasets stored in Amazon S3. More information can be found in [the dataset config documentation](docs/datasets.md)

# Todo
- [ ] Add troubleshooting section
- [ ] Add contribution guidelines 
