import json
import os
import random
import re

import ftfy
import tqdm

NQ_TRAIN_FILE = "data/kilt/nq-train-kilt.jsonl"
NQ_DEV_FILE = "data/kilt/nq-dev-kilt.jsonl"

TRAIN_FILE = "data/train_pairs_nq.jsonl"
EVAL_FILE = "data/eval_queries_nq.jsonl"
CORPUS_FILE = "data/corpus_nq.jsonl"

N_TRAIN_PAIRS = 58000
N_EVAL = 200
N_CORPUS = 10000
NGRAM_WORDS = 10
SEED = 42

random.seed(SEED)


def clean(text):
    text = re.sub(r"\s+", " ", text)
    return ftfy.fix_text(text).strip()


def answer_to_ngram(answer, ngram_len=10):
    """
    Use the gold answer text as the n-gram target.
    Truncate to ngram_len words if long; use as-is if short.
    Never pad with question words — that causes the model to
    copy question words instead of generating answer-relevant spans.
    """
    answer = clean(answer)
    a_words = answer.split()
    # Return up to ngram_len words of the answer
    return " ".join(a_words[:ngram_len]) if a_words else None


print(f"Processing NQ train ({NQ_TRAIN_FILE}) …")
train_pairs = []
corpus_items = {}

with open(NQ_TRAIN_FILE) as f:
    for line in tqdm.tqdm(f, total=87372, desc="Train pairs"):
        if len(train_pairs) >= N_TRAIN_PAIRS:
            break
        s = json.loads(line)
        question = clean(s["input"])

        for out in s.get("output", []):
            if len(train_pairs) >= N_TRAIN_PAIRS:
                break
            answer = out.get("answer", "").strip()
            if not answer or not re.search(r"[a-zA-Z]", answer):
                continue

            prov = out.get("provenance", [{}])[0]
            title = prov.get("title", "Unknown").strip()
            wiki_id = prov.get("wikipedia_id", "")
            para_id = prov.get("start_paragraph_id", 0)
            doc_key = f"{wiki_id}-{para_id}"

            if doc_key and doc_key not in corpus_items and len(corpus_items) < N_CORPUS:
                corpus_items[doc_key] = {
                    "id": f"wiki_{len(corpus_items) + 1:06d}",
                    "title": title,
                    "text": f"{title}. {answer}",
                    "kb_key": doc_key,
                }

            ng = answer_to_ngram(answer, NGRAM_WORDS)
            if ng and len(ng) > 3 and re.search(r"[a-zA-Z]{3,}", ng):
                train_pairs.append(
                    {
                        "input": " " + question,
                        "target": " " + ng,
                    }
                )

random.shuffle(train_pairs)
os.makedirs("data", exist_ok=True)
with open(TRAIN_FILE, "w") as f:
    for p in train_pairs:
        f.write(json.dumps(p) + "\n")
print(f"  ✓ {len(train_pairs):,} training pairs → {TRAIN_FILE}")

corpus = list(corpus_items.values())
with open(CORPUS_FILE, "w") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")
print(f"  ✓ {len(corpus):,} corpus entries → {CORPUS_FILE}")


print(f"\nProcessing NQ dev ({NQ_DEV_FILE}) …")
corpus_key_to_id = {doc["kb_key"]: doc["id"] for doc in corpus}
eval_queries = []
seen = set()

with open(NQ_DEV_FILE) as f:
    for line in tqdm.tqdm(f, desc="Eval queries"):
        if len(eval_queries) >= N_EVAL:
            break
        s = json.loads(line)
        question = clean(s["input"])
        if question in seen:
            continue

        for out in s.get("output", []):
            prov = out.get("provenance", [{}])
            if not prov:
                continue
            p = prov[0]
            wiki_id = p.get("wikipedia_id", "")
            para_id = p.get("start_paragraph_id", 0)
            doc_key = f"{wiki_id}-{para_id}"
            if doc_key in corpus_key_to_id:
                eval_queries.append(
                    {
                        "question": question,
                        "gold_id": corpus_key_to_id[doc_key],
                    }
                )
                seen.add(question)
                break

if len(eval_queries) < 50:
    print(
        f"  Only {len(eval_queries)} dev queries matched corpus; adding train queries …"
    )
    with open(NQ_TRAIN_FILE) as f:
        for line in tqdm.tqdm(f, desc="Train eval fallback"):
            if len(eval_queries) >= N_EVAL:
                break
            s = json.loads(line)
            question = clean(s["input"])
            if question in seen:
                continue
            for out in s.get("output", []):
                prov = out.get("provenance", [{}])
                if not prov:
                    continue
                p = prov[0]
                doc_key = (
                    f"{p.get('wikipedia_id', '')}-{p.get('start_paragraph_id', 0)}"
                )
                if doc_key in corpus_key_to_id:
                    eval_queries.append(
                        {
                            "question": question,
                            "gold_id": corpus_key_to_id[doc_key],
                        }
                    )
                    seen.add(question)
                    break

with open(EVAL_FILE, "w") as f:
    for q in eval_queries:
        f.write(json.dumps(q) + "\n")
print(f"{len(eval_queries)} eval queries → {EVAL_FILE}")

print(f"\n{'=' * 60}")
print(f"  NQ Data Preparation Complete")
print(f"{'=' * 60}")
print(f"  Source        : KILT Natural Questions (real Google queries)")
print(f"  Train pairs   : {len(train_pairs):,}  ({TRAIN_FILE})")
print(f"  Corpus        : {len(corpus):,} Wikipedia passages  ({CORPUS_FILE})")
print(f"  Eval queries  : {len(eval_queries)}  ({EVAL_FILE})")
print(f"  Target type   : Gold NQ answer → {NGRAM_WORDS}-word window")
print(f"{'=' * 60}")
print(f"\nRun fine-tuning:")
print(f"  python3 scripts/finetune_seal.py \\")
print(f"    --train_file {TRAIN_FILE} \\")
print(f"    --epochs 10 --batch_size 8 --output_dir models/seal_nq")
