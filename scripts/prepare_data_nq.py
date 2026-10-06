#!/usr/bin/env python3
"""
Step 1 (v3 — Paper-faithful): Prepare corpus and fine-tuning data using
the KILT Natural Questions dataset + Wikipedia passage knowledge base.

This replicates the paper's exact training data construction:
  - Source: KILT NQ training set (questions + Wikipedia gold provenance)
  - Target: Word-level spans from the gold passage, biased toward
            words overlapping with the query (paper's extract_spans method)
  - N-gram length: 10 tokens (word-level, matching paper)
  - Corpus: Wikipedia 100-word passages (DPR split, psgs_w100.tsv)

Downloads required (auto-handled):
  data/kilt/nq-train-kilt.jsonl   (~50MB)
  data/kilt/nq-dev-kilt.jsonl     (~5MB)
  data/kilt/psgs_w100.tsv.gz      (~5GB uncompressed)

Saves:
    data/corpus_wiki.jsonl         — Wikipedia passage corpus (subset)
    data/train_pairs_nq.jsonl      — (question, n-gram) training pairs
    data/eval_queries_nq.jsonl     — (question, gold_id) eval pairs
"""

import collections
import gzip
import json
import math
import os
import random
import re
import sys

import ftfy
import tqdm
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
NQ_TRAIN_FILE  = "data/kilt/nq-train-kilt.jsonl"
NQ_DEV_FILE    = "data/kilt/nq-dev-kilt.jsonl"
WIKI_TSV_GZ    = "data/kilt/psgs_w100.tsv.gz"
WIKI_TSV       = "data/kilt/psgs_w100.tsv"

CORPUS_FILE    = "data/corpus_wiki.jsonl"
TRAIN_FILE     = "data/train_pairs_nq.jsonl"
EVAL_FILE      = "data/eval_queries_nq.jsonl"

N_TRAIN_PAIRS  = 58000    # matches paper
N_EVAL         = 500      # dev eval queries
N_CORPUS       = 50000    # Wikipedia passages to keep for FM-Index
                          # (use all 21M for paper-level results)
NGRAM_LEN      = 10       # fixed word-level n-gram length (paper)
N_SAMPLES      = 1        # span samples per (question, passage) pair
TEMPERATURE    = 1.0
SEED           = 42

random.seed(SEED)
STOPWORDS = set(stopwords.words('english'))

# ── Helpers (matching paper's make_supervised_kilt_dataset.py) ───────────────
def clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = ftfy.fix_text(text)
    text = text.replace('BULLET::::', '').replace('SECTION::::', '')
    return text.strip()

def preprocess_question(q):
    q = q.strip().replace('\n', ' / ')
    return re.sub(r'\s+', ' ', q)

def span_iterator(tokens, ngrams=3):
    for i in range(len(tokens)):
        if tokens[i].lower() not in STOPWORDS:
            yield (i, i + ngrams)

def extract_span(text, source, min_len, max_len, temperature=1.0):
    """
    Paper's exact span extraction: find word positions in the passage
    that are most lexically similar to the query, then sample a window.
    """
    source_clean = source.split("||", 1)[0].strip()
    q_tokens  = source_clean.split()
    q_lower   = [t.lower() for t in q_tokens]
    p_tokens  = text.split()
    p_lower   = [t.lower() for t in p_tokens]

    matches = collections.defaultdict(float)
    for i1 in range(len(q_lower)):
        j1   = i1 + 3
        str1 = " ".join(q_lower[i1:j1])
        for i2, j2 in span_iterator(p_lower, 3):
            str2  = " ".join(p_lower[i2:j2])
            ratio = fuzz.ratio(str1, str2) / 100.0
            matches[i2] += ratio

    if not matches:
        indices, weights = [0], [1.0]
    else:
        indices, weights = zip(*sorted(matches.items(), key=lambda x: -x[1]))
        weights = [math.exp(w / temperature) for w in weights]
        Z = sum(weights)
        weights = [w / Z for w in weights]

    idx = random.choices(list(indices), weights=list(weights), k=1)[0]
    size = random.randint(min_len, max_len)
    span = " ".join(p_tokens[idx: idx + size])
    return span if re.search(r"[a-zA-Z]", span) else None

# ── Load Wikipedia KB ─────────────────────────────────────────────────────────
def load_wiki_kb(tsv_gz_path, tsv_path):
    """Load passage KB: {wiki_id-para_id → (title, text)}"""
    # Decompress if needed
    if not os.path.exists(tsv_path):
        print(f"Decompressing {tsv_gz_path} (this takes a few minutes) …")
        with gzip.open(tsv_gz_path, 'rt', encoding='utf-8') as fin, \
             open(tsv_path, 'w', encoding='utf-8') as fout:
            for line in tqdm.tqdm(fin, desc="Decompressing"):
                fout.write(line)
        print("  ✓ Decompressed")

    print("Loading Wikipedia passage KB …")
    kb = {}
    with open(tsv_path, encoding='utf-8') as f:
        next(f)  # skip header
        for line in tqdm.tqdm(f, desc="Loading KB"):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 2)
            if len(parts) < 3:
                continue
            idx, title, text = parts
            text = text.replace('BULLET::::', '').replace('Section::::', '')
            text = re.sub(r'\s+', ' ', text).strip()
            if idx and title and text:
                kb[idx] = (title.strip(), text)
    print(f"  ✓ KB loaded: {len(kb):,} passages")
    return kb

# ── Process KILT NQ ───────────────────────────────────────────────────────────
def process_kilt(nq_file, kb, max_pairs, n_samples=1):
    """Generate (question, n-gram) pairs from KILT NQ + Wikipedia KB."""
    pairs = []
    skipped = 0

    with open(nq_file) as f:
        for line in tqdm.tqdm(f, desc="Processing NQ"):
            if len(pairs) >= max_pairs:
                break
            sample = json.loads(line)
            question = preprocess_question(sample['input'])
            source   = " " + question

            for out in sample.get('output', []):
                if len(pairs) >= max_pairs:
                    break
                for prov in out.get('provenance', []):
                    if len(pairs) >= max_pairs:
                        break
                    wiki_id  = prov.get('wikipedia_id', '')
                    para_id  = prov.get('start_paragraph_id', 0)
                    kb_key   = f"{wiki_id}-{para_id}"

                    if kb_key not in kb:
                        skipped += 1
                        continue

                    title, text = kb[kb_key]
                    text = clean(text)

                    for _ in range(n_samples):
                        span = extract_span(text, source, NGRAM_LEN, NGRAM_LEN, TEMPERATURE)
                        if span:
                            pairs.append({"input": source, "target": " " + span.strip()})

    print(f"  Generated {len(pairs):,} pairs  (skipped {skipped:,} missing KB entries)")
    return pairs

# ── Build corpus subset from KB ───────────────────────────────────────────────
def build_corpus(kb, n):
    """Sample N passages from the KB for the retrieval corpus."""
    keys   = list(kb.keys())
    random.shuffle(keys)
    corpus = []
    for i, key in enumerate(keys[:n]):
        title, text = kb[key]
        corpus.append({"id": f"wiki_{i+1:06d}", "title": title, "text": text, "kb_key": key})
    return corpus

# ── Build eval queries from KILT dev set ─────────────────────────────────────
def build_eval(nq_dev_file, kb, corpus, n):
    """
    Build eval queries: (question, gold_id) where gold_id is the corpus
    passage that contains the gold provenance.
    """
    corpus_key_to_id = {doc["kb_key"]: doc["id"] for doc in corpus}
    eval_queries = []
    seen = set()

    with open(nq_dev_file) as f:
        for line in tqdm.tqdm(f, desc="Building eval"):
            if len(eval_queries) >= n:
                break
            sample = json.loads(line)
            question = preprocess_question(sample['input'])
            if question in seen:
                continue

            for out in sample.get('output', []):
                for prov in out.get('provenance', []):
                    wiki_id = prov.get('wikipedia_id', '')
                    para_id = prov.get('start_paragraph_id', 0)
                    kb_key  = f"{wiki_id}-{para_id}"
                    if kb_key in corpus_key_to_id:
                        eval_queries.append({
                            "question": question,
                            "gold_id":  corpus_key_to_id[kb_key],
                        })
                        seen.add(question)
                        break

    return eval_queries

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Verify downloads
    for fpath, name in [(NQ_TRAIN_FILE, "NQ train"), (NQ_DEV_FILE, "NQ dev")]:
        if not os.path.exists(fpath):
            print(f"ERROR: {name} not found at {fpath}")
            print(f"Please download: wget http://dl.fbaipublicfiles.com/KILT/{os.path.basename(fpath)} -O {fpath}")
            sys.exit(1)

    wiki_available = os.path.exists(WIKI_TSV_GZ) or os.path.exists(WIKI_TSV)
    if not wiki_available:
        print("ERROR: Wikipedia KB not found.")
        print(f"Please download: wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz -O {WIKI_TSV_GZ}")
        sys.exit(1)

    # Load KB
    kb = load_wiki_kb(WIKI_TSV_GZ, WIKI_TSV)

    # Build corpus subset
    print(f"\nBuilding corpus ({N_CORPUS:,} passages) …")
    corpus = build_corpus(kb, N_CORPUS)
    os.makedirs("data", exist_ok=True)
    with open(CORPUS_FILE, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")
    print(f"  ✓ Corpus saved: {len(corpus):,} passages → {CORPUS_FILE}")

    # Generate training pairs (paper's exact span extraction)
    print(f"\nGenerating {N_TRAIN_PAIRS:,} training pairs from NQ …")
    train_pairs = process_kilt(NQ_TRAIN_FILE, kb, N_TRAIN_PAIRS, N_SAMPLES)
    random.shuffle(train_pairs)
    with open(TRAIN_FILE, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  ✓ Training pairs saved: {len(train_pairs):,} → {TRAIN_FILE}")

    # Build eval queries
    print(f"\nBuilding eval queries from NQ dev …")
    eval_queries = build_eval(NQ_DEV_FILE, kb, corpus, N_EVAL)
    if not eval_queries:
        # Fallback: use training questions with gold in corpus
        print("  No dev queries matched corpus; falling back to train queries …")
        eval_queries = build_eval(NQ_TRAIN_FILE, kb, corpus, N_EVAL)
    with open(EVAL_FILE, "w") as f:
        for q in eval_queries:
            f.write(json.dumps(q) + "\n")
    print(f"  ✓ Eval queries saved: {len(eval_queries)} → {EVAL_FILE}")

    print(f"\n{'='*60}")
    print(f"  Data preparation complete (Paper-faithful)")
    print(f"{'='*60}")
    print(f"  Dataset       : KILT Natural Questions")
    print(f"  Corpus        : {len(corpus):,} Wikipedia passages")
    print(f"  Train pairs   : {len(train_pairs):,} (paper method, 10-tok spans)")
    print(f"  Eval queries  : {len(eval_queries)}")
    print(f"  N-gram target : fixed {NGRAM_LEN} word tokens")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  python3 scripts/finetune_seal.py --train_file {TRAIN_FILE} \\")
    print(f"      --epochs 10 --batch_size 8 --output_dir models/seal_nq")
    print(f"  python3 scripts/reproduce_seal.py --model models/seal_nq \\")
    print(f"      --corpus {CORPUS_FILE} --queries {EVAL_FILE}")

if __name__ == "__main__":
    main()
