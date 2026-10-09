import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from descfuse.data import load_codes, read_records, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    codes = load_codes(config["codes"])
    allowed = {r["code"] for r in codes}
    splits = {name: read_records(path) for name, path in
              [("train", config["train"]), ("validation", config["validation"]), ("test", args.test)]}
    ids, summary = {}, {"candidate_codes": len(codes), "splits": {}}
    for name, rows in splits.items():
        identifiers = [str(row["id"]) for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate note identifiers in {name}.")
        ids[name] = set(identifiers)
        counts = Counter(c for row in rows for c in set(row["labels"]))
        if counts.keys() - allowed:
            raise ValueError(f"Labels outside the candidate taxonomy in {name}.")
        summary["splits"][name] = {"notes": len(rows), "positive_assignments": sum(counts.values()),
                                   "observed_codes": len(counts), "empty_labels": sum(not r["labels"] for r in rows),
                                   "empty_normalized_notes": sum(not tokens(r["text"]) for r in rows),
                                   "overlength_notes": sum(len(tokens(r["text"])) > config["max_note_tokens"] for r in rows)}
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if ids[first] & ids[second]:
            raise ValueError(f"Note identifiers overlap between {first} and {second}.")
    summary["applicable_reference_slots"] = sum(f["supported"] for row in codes for f in row["facets"])
    summary["train_code_frequency"] = dict(Counter(c for row in splits["train"] for c in set(row["labels"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
