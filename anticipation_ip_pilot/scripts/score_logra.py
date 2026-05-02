import torch
from transformers import AutoModelForCausalLM, default_data_collator
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import random
from dattri.algorithm.logra import LoGraAttributor
import torch.nn as nn
from pathlib import Path
import sys

PILOT_ROOT = Path(__file__).resolve().parents[1]
if str(PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_ROOT))

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    # For older versions of transformers
    from transformers.modeling_utils import Conv1D
from anticipation.vocab import AUTOREGRESS


class TextDataset(Dataset):
    def __init__(self, file_path, max_length=1024, num_samples=None, is_generated: bool = False):
        self.examples = []
        self.max_length = max_length
        self.is_generated = is_generated
        
        print(f"Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                self.examples.append(line.strip())
        
        print(f"Loaded {len(self.examples)} samples from {file_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        if self.is_generated:
            # generated_samples.txt: first/last tokens missing → prepend AUTOREGRESS; keep all tokens
            arr = np.fromstring(self.examples[idx], dtype=int, sep=" ")
            if arr.size == 0:
                input_ids = np.array([AUTOREGRESS], dtype=int)
            else:
                input_ids = np.concatenate([np.array([AUTOREGRESS], dtype=int), arr])
        else:
            # standard format: drop last token (file id)
            safe_text = " ".join(self.examples[idx].split()[:-1])
            input_ids = np.fromstring(safe_text, dtype=int, sep=" ")
        input_ids = input_ids[:self.max_length]
        
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones_like(torch.tensor(input_ids, dtype=torch.long))}


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate attribution scores using LoGra.')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training data.')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Path to validation data.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for saving results.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for dataloaders.')
    parser.add_argument('--proj_dim', type=int, default=4096,
                        help='LoGra random projection dimension.')
    parser.add_argument('--damping', type=float, default=0.01,
                        help='LoGra damping parameter.')
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--valid_is_generated', action='store_true',
                        help='If set, treat valid_file as generated samples: prepend AUTOREGRESS and do not drop last token.')
    parser.add_argument('--output_filename', type=str, default='score_LoGra_4096_gen.pt',
                        help='Output filename to save attribution scores.')
    return parser.parse_args()

def find_layers(model, layer_type="Linear", return_type="instance"):
    layers = []
    return_module_name = not (return_type == "instance")

    if return_module_name:
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm) or isinstance(module, nn.Embedding):
                layers.append((module_name, module))
    else:
        for module in model.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm) or isinstance(module, nn.Embedding):
                layers.append(module)

    if return_module_name:
        if layer_type == "Linear":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, (nn.Linear, nn.LayerNorm))]
        elif layer_type == "LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, nn.LayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")
    else:
        if layer_type == "Linear":
            layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm)]
        elif layer_type == "LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, nn.LayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")

    if return_type == "instance":
        return layers
    elif return_type == "name":
        return [name for name, layer in layers]
    elif return_type == "name_instance":
        return [(name, layer) for name, layer in layers]
    else:
        raise ValueError("Invalid return_type. Choose from 'instance', 'name', and 'name_instance'.")

def replace_conv1d_modules(model):
    # GPT-2 is defined in terms of Conv1D. However, this does not work for EK-FAC.
    # Here, we convert these Conv1D modules to linear modules recursively.
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
            )
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
    return model

class FakeAttributionTask:
    def __init__(self, model):
        self._model = model
        self.original_loss_func = self._loss_func

    def get_model(self):
        return self._model

    def _load_checkpoints(self, _ckpt_idx):
        return None

    @staticmethod
    def _loss_func(model, batch, device):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TextDataset(args.train_file)
    # eval_dataset = TextDataset(args.valid_file, num_samples=100)
    eval_dataset = TextDataset(args.valid_file, num_samples=500, is_generated=args.valid_is_generated)
    
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        print(f"Error: Dataset is empty or failed to load. Exiting.")
        return
    
    print(f"The training dataset length: {len(train_dataset)}.")
    print(f"The eval dataset length: {len(eval_dataset)}.")
    
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )

    if not os.path.isdir(args.model_path):
        print(f"Error: Model directory {args.model_path} not found. Exiting.")
        return
    
    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory {args.output_dir} not found. Exiting.")
        return
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="eager").to(device)
    model.eval()
    
    model = replace_conv1d_modules(model)
    layer_names = find_layers(model, "Linear", return_type="name")
    
    task = FakeAttributionTask(model)
    
    attributor = LoGraAttributor(
        task=task,
        layer_names=layer_names,
        hessian="eFIM",
        device=device,
        proj_dim=args.proj_dim,
        offload="cpu",
        damping=args.damping,
    )
    
    print("Caching train dataloader...")
    attributor.cache(train_dataloader)
    
    print("Attributing scores...")
    # with torch.no_grad():
    score = attributor.attribute(train_dataloader, eval_dataloader)
    
    output_score_file = os.path.join(args.output_dir, args.output_filename)
    torch.save(score, output_score_file)
    print(f"Results saved to {output_score_file}")
    print(f"Score shape: {score.shape}")
    
    print("Processing completed.")

if __name__ == "__main__":
    main()
