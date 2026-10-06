import argparse
import json
import math
import os
import random
import time
from typing import List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup

TRAIN_FILE      = "data/train_pairs.jsonl"
MODEL_NAME      = "facebook/bart-large"
OUTPUT_DIR      = "models/seal_finetuned"
MAX_INPUT_LEN   = 64
MAX_TARGET_LEN  = 32


class NGramDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_input: int, max_target: int):
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target
        self.examples: List[Tuple[str, str]] = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append((obj["input"], obj["target"]))
        print(f"  Loaded {len(self.examples)} training pairs from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, max_input, max_target, device):
    inputs  = [ex[0] for ex in batch]
    targets = [ex[1] for ex in batch]

    enc = tokenizer(
        inputs,
        max_length=max_input,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        dec = tokenizer(
            targets,
            max_length=max_target,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    labels = dec["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids":      enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels":         labels.to(device),
    }


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  SEAL Fine-tuning (PyTorch)")
    print(f"  Base model  : {args.model_name}")
    print(f"  Train file  : {args.train_file}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}  (grad accum: {args.grad_accum})")
    print(f"  LR          : {args.lr}")
    print(f"  Device      : {device}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    use_fp16 = torch.cuda.is_available()
    scaler   = torch.cuda.amp.GradScaler() if use_fp16 else None

    dataset = NGramDataset(args.train_file, tokenizer, args.max_input_len, args.max_target_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_input_len, args.max_target_len, device),
        num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps   = math.ceil(len(loader) / args.grad_accum) * args.epochs
    warmup_steps  = max(1, int(0.1 * total_steps))
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n  Training steps : {total_steps}  (warmup: {warmup_steps})")
    print(f"  Batches/epoch  : {len(loader)}")
    print()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch    = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out  = model(**batch)
                loss = out.loss / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  Step {global_step:4d} | Loss {epoch_loss/(step+1):.4f} | LR {lr_now:.2e}")

        elapsed = time.time() - t_epoch
        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch}/{args.epochs} done — avg loss: {avg_loss:.4f} ({elapsed:.0f}s)\n")

    print(f"Saving to {args.output_dir} …")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved at {args.output_dir}")
    print("\nFine-tuning complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    default=MODEL_NAME)
    parser.add_argument("--train_file",    default=TRAIN_FILE)
    parser.add_argument("--output_dir",    default=OUTPUT_DIR)
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--grad_accum",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=3e-5)
    parser.add_argument("--max_input_len", type=int,   default=MAX_INPUT_LEN)
    parser.add_argument("--max_target_len",type=int,   default=MAX_TARGET_LEN)
    parser.add_argument("--log_every",     type=int,   default=10)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
