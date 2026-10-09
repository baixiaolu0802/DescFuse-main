import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from descfuse import FACETS


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [{"id": row.get("HADM_ID", row.get("id", str(i))),
                 "text": row.get("TEXT", row.get("text", "")),
                 "labels": row.get("LABELS", row.get("labels", "")).split(";")}
                for i, row in enumerate(rows)]
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array, JSONL records, or a CAML-style CSV.")
    return rows


def load_codes(path):
    codes = read_records(path)
    if not codes:
        raise ValueError("The candidate code set must be nonempty.")
    ids = [row["code"] for row in codes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate code identifiers.")
    for row in codes:
        if not isinstance(row["code"], str) or not row.get("description", "").strip():
            raise ValueError("Codes must be strings with nonempty canonical descriptions.")
        records = row["facets"]
        if len(records) != len(FACETS) or {r["name"] for r in records} != set(FACETS):
            raise ValueError(f"Six distinct facets required for {row['code']}.")
        row["facets"] = sorted(records, key=lambda r: FACETS.index(r["name"]))
        for facet in row["facets"]:
            if not isinstance(facet.get("supported"), bool):
                raise ValueError("Facet support must be a boolean.")
            if facet["supported"] != bool(facet.get("description", "").strip()):
                raise ValueError("Unsupported facets must be empty; supported facets must contain text.")
            if facet["supported"] and not facet.get("source_ids"):
                raise ValueError("Supported facets require source identifiers.")
            if not facet["supported"] and facet.get("source_ids"):
                raise ValueError("Unsupported facets must have empty source_ids.")
        if not any(f["supported"] for f in row["facets"]):
            raise ValueError(f"At least one applicable facet required for {row['code']}.")
    return codes


def build_vocab(rows, min_frequency=1, max_size=100000):
    counts = Counter(t for row in rows for t in tokens(row["text"]))
    words = sorted(counts, key=lambda w: (-counts[w], w))
    return {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in
            enumerate([w for w in words if counts[w] >= min_frequency][:max_size - 2])}}


class NoteDataset(Dataset):
    def __init__(self, rows, vocab, codes, max_tokens=4000):
        self.rows, self.vocab, self.max_tokens = rows, vocab, max_tokens
        self.code_to_id = {row["code"]: i for i, row in enumerate(codes)}
        for row in rows:
            if not isinstance(row.get("labels"), list):
                raise ValueError("Each note must contain a labels list; use [] for unlabeled inference.")
            unknown = set(row["labels"]) - self.code_to_id.keys()
            if unknown:
                raise ValueError(f"Unknown labels {unknown}; filter an official subset explicitly.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ids = [self.vocab.get(t, 1) for t in tokens(row["text"])[:self.max_tokens]] or [1]
        labels = torch.zeros(len(self.code_to_id))
        for code in row["labels"]:
            labels[self.code_to_id[code]] = 1
        return torch.tensor(ids), labels, str(row.get("id", index))


def collate_notes(items):
    ids, labels, names = zip(*items)
    lengths = torch.tensor([len(x) for x in ids])
    padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    return {"input_ids": padded, "mask": mask, "labels": torch.stack(labels), "ids": names}
