#!/usr/bin/env python3
"""
Step 3: Run SEAL retrieval evaluation.

Evaluates all three SEAL scoring formulations:
  - LM           : plain beam-search log-prob
  - LM+FM        : log-odds rescoring
  - LM+FM (Inter): intersective additive aggregation

Usage (vanilla model, mini corpus):
    python3 scripts/reproduce_seal.py

Usage (fine-tuned model, large corpus from SQuAD):
    python3 scripts/reproduce_seal.py \
        --corpus  data/corpus.jsonl \
        --queries data/eval_queries.jsonl \
        --model   models/seal_finetuned \
        --beam    15
"""

import argparse
import json
import math
import os
import time
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from seal import FMIndex, fm_index_generate

# English stopwords to filter from beam outputs
STOPWORDS_STR = {
    "the", "a", "an", "in", "of", "to", "and", "or", "for", "with",
    "on", "at", "by", "from", "as", "is", "was", "are", "were", "be",
    "it", "its", "this", "that", "he", "she", "they", "we", "who",
    "which", "what", "how", "when", "where", "not", "but", "if",
    "two", "one", "all", "also", "more", "have", "has", "had",
}

# ── Fallback mini corpus (used when --corpus not provided) ───────────────────
SAMPLE_CORPUS = [
    {"id": "doc_101", "title": "Chopsticks and Cambodian Etiquette",
     "text": "Chopsticks are similar across East Asia, but finer points can differ from region to region. In Cambodia, a fork and spoon are the typical utensils. Chopsticks are normally used in noodle dishes such as Kuy Tiev. When eating soup the chopsticks will typically be paired with the spoon. Forks are never to touch the mouth."},
    {"id": "doc_102", "title": "Earthquake Prediction and Seismology",
     "text": "Seismology is the study of earthquakes and seismic waves. Most seismologists do not believe a system to provide timely warnings for individual earthquakes has yet been developed. Forecasts estimate the probability of an earthquake of a particular magnitude occurring in a given area over a span of years."},
    {"id": "doc_103", "title": "Carbon Footprint and Greenhouse Gases",
     "text": "Carbon dioxide is released naturally by decomposition, ocean release and respiration. Humans contribute an increase of carbon dioxide emissions by burning fossil fuels, deforestation, and cement production. Methane (CH4) is largely released by coal, oil, and natural gas industries."},
    {"id": "doc_104", "title": "Artificial Intelligence and Information Retrieval",
     "text": "Information retrieval systems search for documents and metadata. Autoregressive language models generate n-grams to retrieve relevant context from large collections. FM-Index constrained decoding eliminates vocabulary mismatch and hallucinated document references."},
    {"id": "doc_105", "title": "History of Volcanology",
     "text": "Volcanoes and seismic events have shaped human history. Volcanic eruptions release magma, ash, and gases. Predicting volcanic eruptions and earthquakes remains a complex challenge in geophysics."},
]

SAMPLE_QUERIES = [
    {"question": "can you eat soup with a fork",                   "gold_id": "doc_101"},
    {"question": "can you predict earthquakes",                    "gold_id": "doc_102"},
    {"question": "what causes carbon dioxide emissions",           "gold_id": "doc_103"},
    {"question": "how do autoregressive models retrieve documents","gold_id": "doc_104"},
]

DELIM = "@@"


# ── FM-Index ─────────────────────────────────────────────────────────────────
def build_fm_index(corpus: List[Dict], tokenizer) -> tuple:
    print("\nBuilding FM-Index …")
    t0 = time.time()
    sequences, labels, total_tokens = [], [], 0
    for item in corpus:
        text = f" {item['title']} {DELIM} {item['text']}".strip()
        ids  = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        sequences.append(ids)
        labels.append(item["id"])
        total_tokens += len(ids)
    index = FMIndex()
    index.initialize(sequences, in_memory=False)
    index.labels = labels
    print(f"  ✓ {len(labels)} docs | {total_tokens} tokens | {time.time()-t0:.2f}s")
    return index, total_tokens


def is_degenerate(token_ids, decoded_text, total_tokens, fm_count, n_docs):
    """Filter stopword-only or overly-common n-grams."""
    text = decoded_text.strip().lower()
    # Too short
    if len(text) < 3:
        return True
    # Single stopword
    if text in STOPWORDS_STR:
        return True
    # Appears in >15% of corpus docs (too common to be discriminative)
    if n_docs > 0 and fm_count / total_tokens > 0.015:
        return True
    # No alphabetic content
    if not re.search(r'[a-zA-Z]{2,}', text):
        return True
    return False


# ── Scoring helpers ───────────────────────────────────────────────────────────
def unconditional_prob(index: FMIndex, token_ids: List[int], total: int) -> float:
    count = index.get_count(list(token_ids))
    return count / total if count > 0 else 1e-12


def log_odds(p_nq: float, p_n: float) -> float:
    """SEAL LM+FM formula (§3.2)."""
    p_nq = max(1e-7, min(1 - 1e-7, p_nq))
    p_n  = max(1e-12, min(1 - 1e-7, p_n))
    return max(0.0, math.log((p_nq * (1 - p_n)) / (p_n * (1 - p_nq))))


# ── Retrieval evaluation ──────────────────────────────────────────────────────
def run_experiment(model, tokenizer, index: FMIndex, total_tokens: int,
                   eval_queries: List[Dict], beam_size: int,
                   alpha: float = 2.0, beta: float = 0.8):

    device = next(model.parameters()).device
    MODES  = ["LM", "LM+FM", "LM+FM (Intersective)"]
    summary = {}

    for mode in MODES:
        print(f"\n{'='*64}")
        print(f"  Scoring: {mode}")
        print(f"{'='*64}")

        hits1 = hits3 = 0

        for qi, sample in enumerate(eval_queries):
            query   = sample.get("question") or sample.get("query", "")
            gold_id = sample["gold_id"]
            t0 = time.time()

            enc = tokenizer(" " + query, return_tensors="pt").to(device)

            # Constrained beam search → List[List[(score, token_list)]]
            beams_batch = fm_index_generate(
                model, index,
                **enc,
                min_length=3,
                max_length=20,
                num_beams=beam_size,
                always_allow_eos=True,
                keep_history=True,
                transformers_output=False,
            )
            beams = beams_batch[0]   # batch size = 1

            doc_scores = {did: 0.0 for did in index.labels}
            doc_ngrams  = {did: [] for did in index.labels}

            for lm_score, token_ids in beams:
                valid = [
                    t for t in token_ids
                    if t not in (tokenizer.bos_token_id,
                                 tokenizer.eos_token_id,
                                 tokenizer.pad_token_id)
                    and t is not None
                ]
                if not valid or index.get_count(valid) == 0:
                    continue

                p_nq  = math.exp(max(-30.0, lm_score))
                p_n   = unconditional_prob(index, valid, total_tokens)
                fm_cnt = index.get_count(valid)
                text  = tokenizer.decode(valid, skip_special_tokens=True).strip()

                # Skip degenerate (stopword / too-common) n-grams
                if is_degenerate(valid, text, total_tokens, fm_cnt, len(index.labels)):
                    continue

                score = p_nq if mode == "LM" else log_odds(p_nq, p_n)

                for doc_idx in set(index.get_doc_indices(valid)):
                    if doc_idx >= len(index.labels):
                        continue
                    did = index.labels[doc_idx]
                    doc_ngrams[did].append((text, index.get_count(valid), score))
                    if mode in ("LM", "LM+FM"):
                        doc_scores[did] = max(doc_scores[did], score)
                    else:
                        doc_scores[did] += (score ** alpha) * beta

            ranked  = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            elapsed = (time.time() - t0) * 1000
            top_id  = ranked[0][0] if ranked else "None"
            top_sc  = ranked[0][1] if ranked else 0.0
            top3    = [r[0] for r in ranked[:3]]
            h1      = top_id == gold_id
            h3      = gold_id in top3
            if h1: hits1 += 1
            if h3: hits3 += 1

            print(f"\n  Q{qi+1}: \"{query}\"")
            print(f"  Gold={gold_id}  Top-1={top_id}  (score={top_sc:.4f})  [{elapsed:.0f}ms]")
            if doc_ngrams[top_id]:
                for ng, freq, sc in sorted(doc_ngrams[top_id], key=lambda x: -x[2])[:3]:
                    print(f"    • \"{ng}\"  (FM={freq}, score={sc:.4f})")
            print(f"  H@1={'✓' if h1 else '✗'}  H@3={'✓' if h3 else '✗'}")

        n  = len(eval_queries)
        a1 = hits1 / n * 100
        a3 = hits3 / n * 100
        summary[mode] = (hits1, hits3, a1, a3)
        print(f"\n  → {mode}:  H@1={hits1}/{n} ({a1:.1f}%)  H@3={hits3}/{n} ({a3:.1f}%)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  SEAL RESULTS SUMMARY")
    print(f"{'='*64}")
    print(f"  {'Scoring':<28} {'H@1':>5} {'H@1%':>7} {'H@3':>5} {'H@3%':>7}")
    print(f"  {'-'*52}")
    for mode, (h1, h3, a1, a3) in summary.items():
        print(f"  {mode:<28} {h1:>5} {a1:>6.1f}% {h3:>5} {a3:>6.1f}%")
    print(f"{'='*64}\n")
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="facebook/bart-large",
                        help="Model name or path to fine-tuned checkpoint")
    parser.add_argument("--corpus",     default=None,
                        help="Path to corpus JSONL (id, title, text)")
    parser.add_argument("--queries",    default=None,
                        help="Path to eval queries JSONL (question, gold_id)")
    parser.add_argument("--beam",       type=int, default=15)
    parser.add_argument("--max_corpus", type=int, default=None,
                        help="Cap corpus size to this many passages")
    parser.add_argument("--max_queries",type=int, default=None,
                        help="Cap number of eval queries")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load corpus
    if args.corpus and os.path.exists(args.corpus):
        with open(args.corpus) as f:
            corpus = [json.loads(l) for l in f]
        if args.max_corpus and len(corpus) > args.max_corpus:
            # Keep all gold-relevant passages first, then fill with random
            corpus = corpus[:args.max_corpus]
            print(f"Corpus capped to {len(corpus)} docs (--max_corpus {args.max_corpus})")
        else:
            print(f"Corpus loaded: {len(corpus)} docs from {args.corpus}")
    else:
        corpus = SAMPLE_CORPUS
        print(f"Using built-in mini corpus ({len(corpus)} docs)")

    # Load eval queries
    if args.queries and os.path.exists(args.queries):
        with open(args.queries) as f:
            eval_queries = [json.loads(l) for l in f]
        if args.max_queries and len(eval_queries) > args.max_queries:
            eval_queries = eval_queries[:args.max_queries]
        print(f"Eval queries loaded: {len(eval_queries)} from {args.queries}")
    else:
        eval_queries = SAMPLE_QUERIES
        print(f"Using built-in sample queries ({len(eval_queries)} queries)")

    print(f"\n{'='*64}")
    print(f"  SEAL Retrieval Experiment")
    print(f"  Model  : {args.model}")
    print(f"  Corpus : {len(corpus)} passages")
    print(f"  Queries: {len(eval_queries)}")
    print(f"  Beam   : {args.beam}")
    print(f"  Device : {device}")
    print(f"{'='*64}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()
    print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    index, total_tokens = build_fm_index(corpus, tokenizer)
    run_experiment(model, tokenizer, index, total_tokens, eval_queries,
                   beam_size=args.beam)


if __name__ == "__main__":
    main()
