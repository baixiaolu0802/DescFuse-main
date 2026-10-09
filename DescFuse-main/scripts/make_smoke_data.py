import argparse
import json
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from descfuse.data import load_codes, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/smoke_fixture")
    args = parser.parse_args()
    torch.manual_seed(17)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    codes = load_codes("sample/codes.json")[:2]
    texts = ["Cough fever and chest infiltrate treated with antibacterial therapy.",
             "Bilateral reduced vocal fold movement with hoarse voice and noisy breathing.",
             "Pneumonia with low oxygen and productive cough.",
             "Endoscopy confirms partial bilateral vocal cord paralysis with swallowing difficulty.",
             "Cough and fever with chest opacity.", "Hoarse voice and bilateral vocal fold movement impairment."]
    rows = [{"id": f"synthetic-{i}", "text": text, "labels": [codes[i % 2]["code"]],
             "provenance": "synthetic_software_test"} for i, text in enumerate(texts)]
    for split, subset in [("train", rows[:4]), ("validation", rows[4:])]:
        (root / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in subset), encoding="utf-8")
    (root / "codes.json").write_text(json.dumps(codes, indent=2), encoding="utf-8")
    corpus = texts + [r["description"] for r in codes] + [f["description"] for r in codes for f in r["facets"]]
    words = sorted(set(t for text in corpus for t in tokens(text)))
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[EOS]", *words])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    base = root / "base_model"
    tokenizer.save_pretrained(base)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                max_position_embeddings=512, pad_token_id=0, eos_token_id=2)).save_pretrained(base)
    config = {"base_model": str(base), "llm_dtype": "float32", "gradient_checkpointing": False,
              "codes": str(root / "codes.json"), "train": str(root / "train.jsonl"),
              "validation": str(root / "validation.jsonl"), "output": "runs/smoke", "device": "cpu",
              "seed": 17, "cpu_threads": 1, "embedding_dim": 8, "hidden_size": 8, "facet_dim": 16,
              "prediction_hidden": 24, "dropout": 0.0, "lora_rank": 2, "lora_alpha": 32,
              "lora_targets": ["q_proj", "v_proj"], "lora_lr": 0.0002, "interaction_lr": 0.0001,
              "classifier_lr": 0.0005, "description_weight": 0.01, "batch_size": 2,
              "max_note_tokens": 32, "max_prompt_tokens": 32, "max_reference_tokens": 24,
              "facet_micro_batch": 6, "candidate_budget": 2, "inference_block_size": 1,
              "epochs": 2, "patience": 10, "precision_k": 2}
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(root / "config.yaml")


if __name__ == "__main__":
    main()
