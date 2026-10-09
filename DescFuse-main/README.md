# DescFuse

Description-supervised facet-aware evidence fusion for automated ICD coding from long clinical notes.

This repository implements the method in the supplied DescFuse manuscript and supplementary material. It includes shared-LoRA facet learning, reference-description reconstruction, a BiLSTM document encoder, null-aware evidence retrieval, cross-facet interaction, and joint training. Five configurations cover MIMIC-III Full, top-50, rare50, and the MIMIC-IV ICD-9/ICD-10 top-50 settings.

## Installation

Use Python 3.10–3.12. Install a PyTorch 2.6 build appropriate for your CPU/CUDA environment, then install the remaining dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```


## Quick start

```bash
python -m scripts.export_samples
python -m unittest discover -s tests -v
python -m scripts.make_smoke_data
python -m descfuse.cli train --config runs/smoke_fixture/config.yaml
python -m descfuse.cli evaluate --checkpoint runs/smoke/best.pt --data runs/smoke_fixture/validation.jsonl --device cpu --output runs/smoke/evaluation.json
python -m descfuse.cli predict --checkpoint runs/smoke/best.pt --data runs/smoke_fixture/validation.jsonl --device cpu --output runs/smoke/predictions.jsonl
python -m descfuse.cli inspect --checkpoint runs/smoke/best.pt --data runs/smoke_fixture/validation.jsonl --code 482.9 --device cpu --output runs/smoke/evidence.json
python -m descfuse.cli decode --checkpoint runs/smoke/best.pt --code 482.9 --device cpu --output runs/smoke/decoded.json
```


## Data format

Obtain authorized access to MIMIC and UMLS separately. 

Each note is a JSONL record:

```json
{"id":"admission-identifier","text":"discharge summary text","labels":["482.9","428.0"]}
```

Labels and codes are strings; retain periods and leading zeros. For unlabeled prediction, provide `labels: []`. Notes are lowercased, split into alphanumeric tokens, and truncated at the configured note length. The vocabulary is built from training notes only. Unknown words map to `<unk>`. Empty normalized notes receive one unknown token.

Each code has a canonical description and six facet records:

```json
{
  "code": "482.9",
  "description": "Bacterial pneumonia, unspecified",
  "facets": [
    {"name":"Definition","description":"Evidence-supported text.","source_ids":["canonical:482.9"],"supported":true},
    {"name":"Symptoms","description":"","source_ids":[],"supported":false},
    {"name":"Diagnosis","description":"","source_ids":[],"supported":false},
    {"name":"Treatment","description":"","source_ids":[],"supported":false},
    {"name":"Complications","description":"","source_ids":[],"supported":false},
    {"name":"Summary","description":"","source_ids":[],"supported":false}
  ]
}
```

Every code must retain at least one applicable facet. Code-level applicability is fixed across notes; the learned null probability varies with the note. The code loader orders facets consistently and rejects malformed support records. Unknown note labels raise an error instead of disappearing silently.

Convert an existing CAML-style CSV containing `HADM_ID`, `TEXT`, and semicolon-separated `LABELS`:

```bash
python -m scripts.prepare_data --input train_full.csv --output data/private/mimic3_full/train.jsonl
```

For a predefined top-50 label list, supply its code JSON and explicitly request filtering:

```bash
python -m scripts.prepare_data --input train_full.csv --codes top50_codes.json --drop-outside-labels --output data/private/mimic3_50/train.jsonl
```

Use the same official label list across splits. Subset filtering removes notes with no retained labels. It does not select top-50 or rare50 codes, reconstruct official splits, or guarantee the paper's counts. Rare50 requires the original subset definition and split protocol.

## Reference construction

References use taxonomy information and external literature only. Keep patient notes, split assignments, evaluation labels, and predictions out of this pipeline.

1. Export English synonyms and selected clinical relationships from a licensed UMLS release:

```bash
python -m scripts.extract_umls --codes canonical_codes.json --mrconso UMLS/MRCONSO.RRF --mrrel UMLS/MRREL.RRF --icd-source ICD9CM --output data/private/umls.json
```

2. Import a fixed PubMed XML snapshot:

```bash
python -m scripts.import_pubmed --xml pubmed_snapshot.xml.gz --output data/private/abstracts.jsonl
```

3. Build six-facet references with an explicitly selected reference model. The endpoint must accept OpenAI-compatible chat-completions JSON. Set `REFERENCE_API_KEY` in the environment when authentication is required.

```bash
python -m scripts.build_references --codes canonical_codes.json --umls data/private/umls.json --abstracts data/private/abstracts.jsonl --dataset mimic3 --taxonomy ICD9CM-2015 --snapshot YOUR_FIXED_SNAPSHOT --model YOUR_REFERENCE_MODEL --endpoint http://localhost:8000/v1/chat/completions --output data/private/references_icd9
```

The canonical input contains only `code` and `description`. UMLS records contain `synonyms`, `related_concepts`, and `source_ids`. Abstracts contain `id`, `text`, and optionally `title`. BM25 uses k1=1.2 and b=0.75, with three retrieved abstracts by default. The in-memory index is intended for a bounded reference corpus; a complete PubMed collection needs a larger retrieval backend with equivalent settings.

One request produces all six facets for each code. The cache stores evidence, prompts, raw responses, model identity, and a content-based key. Source validation catches absent identifiers; it does not verify that each clinical claim follows from its cited source. Review the references before training. All-unsupported codes fail validation and require evidence review.

Existing responses can be imported without an API request by replacing `--endpoint` with `--responses-dir PATH`. That directory must contain `<code>.json` files with either a `facets` object or a chat-completions response. Reuse parent-taxonomy records for top-50 subsets.

## Training

Place `codes.json`, `train.jsonl`, and `validation.jsonl` under the configured data directory. Keep the candidate code order fixed across runs. Edit the paths in the selected configuration.

Audit candidate coverage, reference applicability, and note-identifier separation before training:

```bash
python -m scripts.audit_dataset --config configs/mimic3_full.yaml --test data/private/mimic3_full/test.jsonl --output runs/mimic3_full/data_audit.json
```

```bash
python -m descfuse.cli train --config configs/mimic3_full.yaml --seed 13 --output runs/mimic3_full/seed13
python -m descfuse.cli train --config configs/mimic3_full.yaml --seed 42 --output runs/mimic3_full/seed42
python -m descfuse.cli train --config configs/mimic3_full.yaml --seed 2026 --output runs/mimic3_full/seed2026
```

The other configurations are `mimic3_50.yaml`, `mimic3_rare50.yaml`, `mimic4_icd9_50.yaml`, and `mimic4_icd10_50.yaml`.

Training retains the union of positive codes in each batch and uniformly samples remaining candidates. Negative weights correct the loss to the full-label BCE objective. The default candidate budget is 128. All positives remain even when they exceed this budget. Queries are recomputed with current parameters every optimization step and remain connected to the classification graph.

`best.pt` stores trainable weights, code records, vocabulary, configuration, the validation-selected threshold, and epoch. `manifest.json` records library versions, trainable parameter count, and hashes of the input files. Frozen base-model weights are loaded from `base_model` when restoring a checkpoint. Keep that exact base-model version available. The checkpoint is for evaluation; optimizer-state resumption is not implemented. Use a fresh output directory for each training run.

The supplied paper configurations use trainable random word embeddings. For pretrained embeddings, export `vocab.json` using `prepare_data --vocab-output`, align a NumPy matrix to that exact vocabulary order, and set `word_embeddings` to its `.npy` path. Use the same frequency and size settings as training.


## Evaluation and evidence inspection

```bash
python -m descfuse.cli evaluate --checkpoint runs/mimic3_full/seed13/best.pt --data data/private/mimic3_full/test.jsonl --output runs/mimic3_full/seed13/test_metrics.json
```

Evaluation scores every candidate in blocks of 256. It applies the stored validation threshold without retuning on test labels. Metrics include Macro-/Micro-F1, Macro-/Micro-AUC, micro average precision, equal-frequency ECE, and P@k. Values are fractions; multiply by 100 for percentages. The adjacent `.npz` contains probabilities, targets, note identifiers, and candidate identifiers for auditing.

```bash
python -m scripts.aggregate_runs --metrics runs/mimic3_full/seed13/test_metrics.json runs/mimic3_full/seed42/test_metrics.json runs/mimic3_full/seed2026/test_metrics.json --output runs/mimic3_full/summary.json
```

`inspect` exports null probabilities, token attention, cross-facet attention, evidence vectors, and pooling weights for one note/code pair. Token indices follow the normalized, truncated note. `decode` reconstructs optional descriptions from the trained continuous prefixes; classification does not depend on decoded text.


```bash
python -m pip install reportlab==4.4.9
python -m scripts.render_samples_pdf
```

`profiles.json` is the editable source. Run `python -m scripts.export_samples` to refresh all formats. Each column contains complete clinical paragraphs across six dimensions. The text and its relationship to checkpoint evaluation are documented in [comparison notes](docs/sample_notes.md). Sources accompany each profile. Code 240 is a category-level entry and requires an appropriate child code for leaf-level experiments.

For a trained checkpoint, collect actual text from both methods:

```bash
python -m scripts.compare_descriptions --checkpoint runs/mimic3_full/seed13/best.pt --codes 482.9 478.33 428.0 493.90 --output runs/mimic3_full/actual_description_comparison.json
```

The comparison disables LoRA on the same frozen backbone for Direct-Prompting and decodes the learned continuous prefixes for DescFuse. Both use greedy decoding. The output retains checkpoint identity and the applicability mask. Review both columns against external evidence before manuscript use.

## Repository layout

```text
configs/        Five benchmark configurations
descfuse/       Model, data, objectives, metrics, training, CLI, reference retrieval
scripts/        Data conversion, UMLS/PubMed import, references, samples, run aggregation
sample/         Five comparison tables and structured reference examples
tests/          Model invariants
```

