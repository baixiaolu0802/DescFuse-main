import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.metrics]
    output = {}
    for key in rows[0]:
        values = [r.get(key) for r in rows]
        if all(isinstance(v, (float, int)) for v in values):
            output[key] = {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                           "runs": len(values)}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
