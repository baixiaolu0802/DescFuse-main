import argparse
import json
from pathlib import Path

from descfuse.data import build_vocab, read_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codes")
    parser.add_argument("--drop-outside-labels", action="store_true")
    parser.add_argument("--vocab-output")
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=100000)
    args = parser.parse_args()
    rows = read_records(args.input)
    allowed = {r["code"] for r in read_records(args.codes)} if args.codes else None
    retained = []
    for row in rows:
        if allowed is not None:
            outside = set(row["labels"]) - allowed
            if outside and not args.drop_outside_labels:
                raise ValueError("Labels outside the candidate set; enable explicit subset filtering.")
            row["labels"] = [c for c in row["labels"] if c in allowed]
            if not row["labels"]:
                continue
        retained.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained), encoding="utf-8")
    if args.vocab_output:
        vocab = build_vocab(retained, min_frequency=args.min_frequency, max_size=args.max_vocab_size)
        Path(args.vocab_output).write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(json.dumps({"input_notes": len(rows), "retained_notes": len(retained)}))


if __name__ == "__main__":
    main()
