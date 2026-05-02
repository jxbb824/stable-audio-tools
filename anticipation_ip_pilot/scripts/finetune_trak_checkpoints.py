import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, TrainerCallback
import numpy as np
from torch.utils.data import Dataset, Subset
import logging
import os
import argparse
from typing import Optional, Tuple
import random
import csv
import json

logging.basicConfig(level=logging.INFO)

# PassthroughTokenizer (must mirror finetune.py)
class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_size, **kwargs):
        self._vocab_size = vocab_size
        self._vocab = {i: i for i in range(self._vocab_size)}
        self._eos = 55025
        self._eos_token = str(self._eos)

        super().__init__(
            eos_token=self._eos_token,
            pad_token=self._eos_token,
            **kwargs
        )
        
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @eos_token.setter
    def eos_token(self, value: str) -> None:
        self._eos_token = str(value)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos

    @eos_token_id.setter
    def eos_token_id(self, value: Optional[int]) -> None:
        if value is not None:
            self._eos = int(value)

    def get_vocab(self):
        return self._vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        safe_text = " ".join(text.split()[:-1]) 
        tokens = np.fromstring(safe_text, dtype=int, sep=" ")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)


class TextDataset(Dataset):
    def __init__(self, file_path, max_length=1024, num_samples=None):
        self.examples = []
        self.max_length = max_length
        
        logging.info(f"Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                self.examples.append(line.strip())
        
        logging.info(f"Loaded {len(self.examples)} samples from {file_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        safe_text = " ".join(self.examples[idx].split()[:-1])
        input_ids = np.fromstring(safe_text, dtype=int, sep=" ")
        input_ids = input_ids[:self.max_length]
        
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(input_ids, dtype=torch.long)}


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 model with TracIn checkpoints (minimal diff).')
    parser.add_argument('--pretrained_model_path', type=str, default='stanford-crfm/music-small-800k',
                        help='Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str, 
                        default='./finetune_subset_output',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating weights')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--subset_ratio', type=float, default=1.0,
                        help='Ratio of training data to randomly select (0.0 to 1.0). Applied to the loaded training data.')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility of subset sampling and training.")
    return parser.parse_args()


class TracinCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.lrs = []

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get('model', None)
        optimizer = kwargs.get('optimizer', None)
        epoch_float = state.epoch if state.epoch is not None else 0.0
        epoch_idx = max(int(epoch_float) - 1, 0)

        ckpt_dir = os.path.join(self.output_dir, str(epoch_idx))
        os.makedirs(ckpt_dir, exist_ok=True)
        if model is not None:
            model.save_pretrained(ckpt_dir)

        lr = None
        if optimizer is not None and len(optimizer.param_groups) > 0:
            lr = optimizer.param_groups[0].get('lr', None)
        if lr is None:
            lr = args.learning_rate
        self.lrs.append(float(lr))

        lrs_path_json = os.path.join(self.output_dir, 'lrs.json')
        with open(lrs_path_json, 'w') as f:
            json.dump({"lrs": self.lrs}, f)

        torch.save(torch.tensor(self.lrs, dtype=torch.float32), os.path.join(self.output_dir, 'lrs.pt'))


def main():
    args = parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Initialize tokenizer (same as finetune.py)
    tokenizer = PassthroughTokenizer(vocab_size=55028)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load pretrained model
    logging.info(f"Loading pretrained model from {args.pretrained_model_path}")
    try:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path).cuda()
        logging.info("Successfully loaded pretrained model")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.warning("Falling back to default configuration")
        model_config = GPT2Config(
            vocab_size=55028,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
        model = GPT2LMHeadModel(model_config)
    
    # Load datasets (same pipeline and subset sampling)
    full_train_dataset = TextDataset(args.train_file)
    
    train_indices_to_save = list(range(len(full_train_dataset)))
    actual_train_dataset = full_train_dataset

    if not (0.0 < args.subset_ratio <= 1.0):
        if args.subset_ratio == 0.0:
            logging.error("subset_ratio is 0.0. No data to train on. Exiting.")
            return
        elif args.subset_ratio > 1.0:
            logging.warning(f"subset_ratio ({args.subset_ratio}) is greater than 1.0. Using full loaded dataset.")
    
    if 0.0 < args.subset_ratio < 1.0:
        if len(full_train_dataset) == 0:
            logging.warning("Full training dataset is empty, cannot create a subset.")
        else:
            num_selected_samples = int(len(full_train_dataset) * args.subset_ratio)
            if num_selected_samples == 0 and len(full_train_dataset) > 0:
                num_selected_samples = 1 
            
            logging.info(f"Attempting to select {num_selected_samples} out of {len(full_train_dataset)} training samples.")

            original_indices = list(range(len(full_train_dataset)))
            selected_indices_for_subset = random.sample(original_indices, num_selected_samples)
            
            actual_train_dataset = Subset(full_train_dataset, selected_indices_for_subset)
            train_indices_to_save = selected_indices_for_subset
            logging.info(f"Randomly selected {len(actual_train_dataset)} samples for training (ratio: {args.subset_ratio}).")
    else:
        logging.info(f"Using full loaded training dataset ({len(actual_train_dataset)} samples).")

    if len(actual_train_dataset) == 0:
        logging.error("No training samples available after subset selection. Exiting.")
        return

    valid_dataset = TextDataset(args.valid_file, num_samples=100) 
    if len(valid_dataset) == 0:
        logging.warning("Validation dataset is empty. Evaluation might not be meaningful.")
        valid_dataset = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="no",  # keep identical; checkpointing is done by our callback
        eval_steps=500,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=500,
        evaluation_strategy="steps" if valid_dataset is not None else "no",
        fp16=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        report_to="none",
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    tracin_callback = TracinCheckpointCallback(args.output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=actual_train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        callbacks=[tracin_callback]
    )
    
    logging.info("Starting fine-tuning (with TracIn checkpoints via callback)...")
    trainer.train()
    
    # Save fine-tuned model to output_dir (same as finetune.py)
    logging.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)

    # Additionally save to output_dir/full_model for scoring pipeline compatibility
    final_dir = os.path.join(args.output_dir, 'full_model')
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    logging.info(f"Saved final model copy to {final_dir}")

    # Save the subset indices (same as finetune.py)
    if train_indices_to_save:
        index_file_path = os.path.join(args.output_dir, 'train_index.csv')
        try:
            with open(index_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for index in train_indices_to_save:
                    writer.writerow([index])
            logging.info(f"Saved training subset indices to {index_file_path}")
        except IOError as e:
            logging.error(f"Could not write training indices to {index_file_path}: {e}")

    logging.info("Fine-tuning with TracIn checkpoints completed!")


if __name__ == "__main__":
    main()
