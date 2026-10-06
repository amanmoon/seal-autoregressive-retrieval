#!/usr/bin/env python3
"""
Step 1 (Improved): Prepare corpus and fine-tuning data from SQuAD.

Improvements over v1:
  - N_CORPUS     : 100  → 500  passages
  - N_TRAIN_PAIRS: 800  → 5000 pairs
  - N_EVAL       : 20   → 50   queries
  - Targets      : random token windows → ANSWER SPANS (gold quality)
                   + random 10-token windows as supplementary negatives
  - N-gram length: variable 6-12 → FIXED 10 tokens (matching paper)

Saves:
    data/corpus.jsonl          — corpus passages
    data/train_pairs.jsonl     — (question, ngram) fine-tuning pairs
    data/eval_queries.jsonl    — (question, gold_id) eval pairs
"""

import json
import random
import re
import sys

from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
SQUAD_FILE      = "data/squad_train.json"
CORPUS_FILE     = "data/corpus.jsonl"
TRAIN_FILE      = "data/train_pairs.jsonl"
EVAL_FILE       = "data/eval_queries.jsonl"

N_CORPUS        = 500     # passages in retrieval corpus
N_EVAL          = 50      # evaluation queries
N_TRAIN_PAIRS   = 5000    # (question, n-gram) training pairs
NGRAM_LEN       = 10      # fixed n-gram length in tokens (matches paper)
ANSWER_RATIO    = 0.6     # fraction of pairs that use gold answer spans
MODEL_NAME      = "facebook/bart-large"
SEED            = 42

random.seed(SEED)

# ── Load SQuAD ────────────────────────────────────────────────────────────────
print("Loading SQuAD …")
with open(SQUAD_FILE) as f:
    squad = json.load(f)["data"]

records = []
for article in squad:
    title = article["title"].replace("_", " ")
    for para in article["paragraphs"]:
        ctx = para["context"].strip()
        qas = [
            {
                "question": qa["question"].strip(),
                "answer":   qa["answers"][0]["text"].strip(),
                "answer_start": qa["answers"][0]["answer_start"],
            }
            for qa in para["qas"]
            if qa["answers"]
        ]
        if qas:
            records.append({"title": title, "context": ctx, "qas": qas})

print(f"  {len(records):,} paragraphs with QA pairs loaded")
random.shuffle(records)

# ── Build corpus ──────────────────────────────────────────────────────────────
corpus_records = records[:N_CORPUS]
corpus = [
    {"id": f"doc_{i+1:04d}", "title": r["title"], "text": r["context"]}
    for i, r in enumerate(corpus_records)
]
with open(CORPUS_FILE, "w") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")
print(f"  ✓ Corpus saved: {len(corpus)} passages → {CORPUS_FILE}")

# ── Build eval queries ────────────────────────────────────────────────────────
print("\nBuilding eval queries …")
eval_queries, seen_titles = [], set()
for i, rec in enumerate(corpus_records):
    if rec["title"] not in seen_titles and rec["qas"]:
        qa = random.choice(rec["qas"])
        eval_queries.append({
            "question": qa["question"],
            "gold_id":  f"doc_{i+1:04d}",
            "gold_title": rec["title"],
        })
        seen_titles.add(rec["title"])
    if len(eval_queries) >= N_EVAL:
        break

with open(EVAL_FILE, "w") as f:
    for q in eval_queries:
        f.write(json.dumps(q) + "\n")
print(f"  ✓ Eval queries: {len(eval_queries)} → {EVAL_FILE}")

# ── Generate training pairs ───────────────────────────────────────────────────
print("\nGenerating training pairs …")
print("  Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def answer_span_ngram(context, answer_text, answer_start, tokenizer, ngram_len):
    """
    Produce a FIXED ngram_len-token n-gram centred around the answer span.
    This is the gold-quality target matching the paper's methodology.
    """
    ids_full  = tokenizer(context, add_special_tokens=False)["input_ids"]
    ans_ids   = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not ans_ids or not ids_full:
        return None

    # Find the answer token start position using char offset
    prefix_ids = tokenizer(context[:answer_start], add_special_tokens=False)["input_ids"]
    ans_tok_start = len(prefix_ids)
    ans_tok_end   = ans_tok_start + len(ans_ids)

    # Centre the ngram window around the answer
    half    = ngram_len // 2
    start   = max(0, ans_tok_start - half)
    start   = min(start, max(0, len(ids_full) - ngram_len))
    end     = min(start + ngram_len, len(ids_full))
    chunk   = ids_full[start:end]

    if len(chunk) < 4:
        return None
    decoded = tokenizer.decode(chunk, skip_special_tokens=True).strip()
    if not re.search(r"[a-zA-Z]", decoded):
        return None
    return decoded


def random_ngram(context, tokenizer, ngram_len):
    """Fallback: sample a random fixed-length token window."""
    ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if len(ids) < ngram_len:
        return None
    start = random.randint(0, len(ids) - ngram_len)
    chunk = ids[start: start + ngram_len]
    decoded = tokenizer.decode(chunk, skip_special_tokens=True).strip()
    if not re.search(r"[a-zA-Z]", decoded) or len(decoded) < 5:
        return None
    return decoded


train_pairs = []
# Use all SQuAD records for training diversity
all_records = records.copy()
random.shuffle(all_records)

n_answer_pairs = int(N_TRAIN_PAIRS * ANSWER_RATIO)
n_random_pairs = N_TRAIN_PAIRS - n_answer_pairs

# --- Gold answer-span pairs ---
print(f"  Generating {n_answer_pairs} answer-span pairs …")
for rec in all_records:
    if len(train_pairs) >= n_answer_pairs:
        break
    for qa in rec["qas"]:
        if len(train_pairs) >= n_answer_pairs:
            break
        ng = answer_span_ngram(
            rec["context"], qa["answer"], qa["answer_start"],
            tokenizer, NGRAM_LEN
        )
        if ng:
            train_pairs.append({"input": " " + qa["question"], "target": ng})

print(f"    → {len(train_pairs)} answer-span pairs generated")

# --- Random 10-token window pairs (supplementary) ---
print(f"  Generating {n_random_pairs} random-window pairs …")
random_pairs = []
random.shuffle(all_records)
for rec in all_records:
    if len(random_pairs) >= n_random_pairs:
        break
    for qa in rec["qas"]:
        if len(random_pairs) >= n_random_pairs:
            break
        ng = random_ngram(rec["context"], tokenizer, NGRAM_LEN)
        if ng:
            random_pairs.append({"input": " " + qa["question"], "target": ng})

train_pairs.extend(random_pairs)
random.shuffle(train_pairs)

with open(TRAIN_FILE, "w") as f:
    for p in train_pairs:
        f.write(json.dumps(p) + "\n")

print(f"\n{'='*60}")
print(f"  Data preparation complete")
print(f"{'='*60}")
print(f"  Corpus          : {len(corpus)} passages  ({CORPUS_FILE})")
print(f"  Train pairs     : {len(train_pairs)} total  ({TRAIN_FILE})")
print(f"    ├─ Answer-span : {len(train_pairs)-len(random_pairs)} ({ANSWER_RATIO*100:.0f}%)")
print(f"    └─ Random 10-tok: {len(random_pairs)} ({(1-ANSWER_RATIO)*100:.0f}%)")
print(f"  Eval queries    : {len(eval_queries)}  ({EVAL_FILE})")
print(f"  N-gram length   : fixed {NGRAM_LEN} tokens (matches paper)")
print(f"{'='*60}")
