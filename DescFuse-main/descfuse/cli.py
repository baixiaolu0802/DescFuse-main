import argparse
import json
from pathlib import Path

import numpy as np
import torch

from descfuse import FACETS
from descfuse.data import read_records
from descfuse.engine import load_checkpoint, make_loader, score_loader, train
from descfuse.metrics import evaluate_metrics


def main():
    parser = argparse.ArgumentParser(prog="descfuse")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--config", required=True)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--output")
    for name in ("evaluate", "predict", "inspect", "decode"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--output", required=True)
        if name != "decode":
            command.add_argument("--data", required=True)
        if name in ("inspect", "decode"):
            command.add_argument("--code", required=True)
        if name == "inspect":
            command.add_argument("--note-index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "train":
        train(args.config, args.seed, args.output)
        return
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    codes, config = checkpoint["codes"], checkpoint["config"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command in ("decode", "inspect"):
        lookup = {row["code"]: i for i, row in enumerate(codes)}
        if args.code not in lookup:
            parser.error(f"Unknown code {args.code}")
        indices = torch.tensor([lookup[args.code]], device=device)
        with torch.no_grad():
            queries, states = model.generator(indices)
            if args.command == "decode":
                descriptions = model.generator.decode(states)[0]
                result = {"code": args.code, "provenance": "checkpoint_decoded",
                          "checkpoint": str(args.checkpoint),
                          "facets": [{"name": name, "description": text} for name, text in zip(FACETS, descriptions)]}
            else:
                rows = read_records(args.data)
                row = rows[args.note_index]
                loader = make_loader([row], checkpoint["vocab"], codes, config)
                batch = next(iter(loader))
                document = model.encoder(batch["input_ids"].to(device), batch["mask"].to(device))
                logits, diagnostics = model.fusion(document, batch["mask"].to(device), queries,
                                                    model.facet_mask[indices], diagnostics=True)
                result = {"id": str(row["id"]), "code": args.code, "facets": list(FACETS),
                          "probability": float(logits.sigmoid()[0, 0]),
                          **{k: v.cpu().tolist() if v is not None else None for k, v in diagnostics.items()}}
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    rows = read_records(args.data)
    loader = make_loader(rows, checkpoint["vocab"], codes, config)
    target, probability, names = score_loader(model, loader, device, config["inference_block_size"])
    if args.command == "evaluate":
        metrics = evaluate_metrics(target, probability, checkpoint["threshold"], config.get("precision_k", 5))
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(output.with_suffix(".npz"), target=target, probability=probability,
                            ids=np.asarray(names), codes=np.asarray([r["code"] for r in codes]))
        print(json.dumps(metrics, indent=2))
    else:
        with output.open("w", encoding="utf-8") as handle:
            for name, scores in zip(names, probability):
                selected = [{"code": row["code"], "probability": float(score)}
                            for row, score in zip(codes, scores) if score >= checkpoint["threshold"]]
                handle.write(json.dumps({"id": name, "predictions": selected}) + "\n")


if __name__ == "__main__":
    main()
