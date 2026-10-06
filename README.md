# SEAL: Search Engines with Autoregressive LMs

**Paper:** *Autoregressive Search Engines: Generating Substrings as
Document Identifiers*\
**Authors:** Bevilacqua et al.\
**arXiv:** 2204.10628

## Overview

SEAL uses an autoregressive language model and an FM-index to perform
retrieval by generating document n-grams directly. The FM-index
constrains generation to spans that actually occur in the corpus, while
n-gram scores combine LM likelihood with corpus frequency.

The presentation covers the SEAL architecture, retrieval mechanism,
comparison with existing retrievers, reproduction results, and
extensions. fileciteturn0file0L31-L43

## Key Components

-   **BART-large:** generates n-grams conditioned on the query.
-   **FM-index:** constrains decoding to corpus-valid n-grams and maps
    them to documents.
-   **Scoring:** combines LM probability and n-gram frequency.
-   **Intersective scoring:** aggregates multiple useful n-grams for
    stronger document ranking. fileciteturn0file0L31-L43

## Reproduction

We reused the authors' pretrained SEAL checkpoint and FM-index, and ran
scaled-down experiments because full-scale fine-tuning and beam search
were too expensive.

Our SQuAD fine-tuning setup used **500 passages, 5,000 training pairs,
10 epochs**, with a final training loss of **1.47**.
fileciteturn1file0

### SQuAD Results

  Method                      Hits@1    Hits@3
  ------------------------ --------- ---------
  LM                             14%       18%
  LM + FM                        32%       52%
  LM + FM (Intersective)     **46%**   **56%**

The experiments reproduce the paper's main trends: LM+FM improves over
LM, intersective scoring performs best, and vanilla BART is ineffective
for retrieval. fileciteturn1file2

## Results

### SQuAD Fine-tuning Experiments

| Configuration | LM H@1 | LM+FM H@1 | Intersective H@1 | Intersective H@3 |
|---|---:|---:|---:|---:|
| Vanilla | 5.0% | 0.0% | 0.0% | 0.0% |
| Fine-tuned v1 | 15.0% | 25.0% | 25.0% | 35.0% |
| Fine-tuned v2 | 14.0% | **32.0%** | **46.0%** | **56.0%** |

For the best v2 configuration, the raw results were:

- **LM:** 7/50 H@1, 9/50 H@3
- **LM+FM:** 16/50 H@1, 26/50 H@3
- **LM+FM (Intersective):** **23/50 H@1, 28/50 H@3**

fileciteturn1file2

### Natural Questions

| Scoring | Paper Acc@5 | Ours Acc@5 | Paper Acc@20 | Ours Acc@20 | Paper Acc@100 | Ours Acc@100 |
|---|---:|---:|---:|---:|---:|---:|
| LM | 40.5 | 35.6 | 60.2 | 62.4 | 73.1 | 78.0 |
| LM+FM | 43.9 | 36.8 | 65.8 | 63.6 | 81.1 | **81.6** |
| LM+FM (Intersective) | 61.3 | **61.2** | 76.2 | **75.2** | 86.3 | **84.4** |

fileciteturn1file5

### KILT

| Task | Passage R-prec (Ours) | Passage R-prec (Paper) | Page R-prec (Ours) | Page R-prec (Paper) |
|---|---:|---:|---:|---:|
| FEVER | **72.0** | 67.8 | **89.0** | 81.4 |
| NQ | **44.0** | 43.6 | 60.0 | **63.2** |
| zsRE | 74.0 | **78.8** | 86.0 | **91.6** |
| **Macro Avg.** | **63.3** | 63.4 | **78.3** | 78.7 |

fileciteturn1file4

## Takeaways

-   SEAL grounds autoregressive retrieval directly in the corpus.
-   FM-index constraints prevent generated evidence from being absent
    from the corpus.
-   Intersective multi-n-gram scoring gives the strongest retrieval
    performance.
-   The scaled-down reproduction successfully matches the main
    qualitative trends of the original work.

## Extensions

The presentation explores three directions:

1.  **Cross-lingual SEAL:** retrieve English evidence from Hindi queries
    or build a multilingual pipeline.
2.  **Structural SEAL:** use document hierarchy for codebases and legal
    contracts.
3.  **SEAL for audio and images:** apply the same indexing idea to
    discrete audio or image representations. fileciteturn1file3

For cross-lingual SEAL, proposed next steps include stronger
translation, native Hindi evaluation, entity-preserving translation,
multilingual generation, a Hindi-side FM-index, and targeted error
analysis. fileciteturn1file1
