import argparse
import json
from pathlib import Path

import torch

from descfuse import FACETS
from descfuse.engine import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--codes", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    lookup = {row["code"]: i for i, row in enumerate(checkpoint["codes"])}
    tokenizer = model.generator.tokenizer
    samples = []
    for code in args.codes:
        index = lookup[code]
        record = checkpoint["codes"][index]
        with torch.no_grad():
            _, states = model.generator(torch.tensor([index], device=device))
            descriptions = model.generator.decode(states, args.max_new_tokens)[0]
            direct = {}
            # 关闭同一骨干上的 LoRA，获得可追溯的直接提示输出。
            with model.generator.llm.disable_adapter():
                for facet in FACETS:
                    prompt = (f"ICD {code}: {record['description']}\n"
                              f"Describe its clinical {facet.lower()} concisely.\nDescription:")
                    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
                    generated = model.generator.llm.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    direct[facet] = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        samples.append({"code": code, "description": record["description"],
                        "provenance": "checkpoint_comparison", "checkpoint": str(args.checkpoint),
                        "base_model": checkpoint["config"]["base_model"], "decoding": "greedy",
                        "max_new_tokens": args.max_new_tokens,
                        "direct_prompting": direct, "descfuse": dict(zip(FACETS, descriptions)),
                        "applicable_facets": {f["name"]: f["supported"] for f in record["facets"]}})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
