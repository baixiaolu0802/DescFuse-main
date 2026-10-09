import json
import random
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from descfuse.data import NoteDataset, build_vocab, collate_notes, load_codes, read_records
from descfuse.metrics import evaluate_metrics, select_threshold
from descfuse.model import DescFuse
from descfuse.objectives import sample_candidates, sampled_bce


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_loader(rows, vocab, codes, config, training=False):
    dataset = NoteDataset(rows, vocab, codes, config["max_note_tokens"])
    return DataLoader(dataset, batch_size=config["batch_size"] if training else config.get("eval_batch_size", 1),
                      shuffle=training, collate_fn=collate_notes, num_workers=0)


@torch.no_grad()
def score_loader(model, loader, device, block_size):
    model.eval()
    targets, probabilities, ids = [], [], []
    for batch in loader:
        moved = move_batch(batch, device)
        probabilities.append(model.predict_all(moved, block_size).float().cpu().numpy())
        targets.append(batch["labels"].numpy())
        ids.extend(batch["ids"])
    if not targets:
        raise ValueError("Empty evaluation split.")
    return np.concatenate(targets), np.concatenate(probabilities), ids


def save_checkpoint(path, model, config, codes, vocab, threshold, epoch):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
             if name in trainable or "lora_" in name or not name.startswith("generator.llm.")}
    torch.save({"state": state, "config": config, "codes": codes, "vocab": vocab,
                "threshold": threshold, "epoch": epoch}, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if device.type == "cpu" and checkpoint["config"].get("cpu_threads"):
        torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    model = DescFuse(checkpoint["config"], checkpoint["codes"], len(checkpoint["vocab"]))
    result = model.load_state_dict(checkpoint["state"], strict=False)
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if result.unexpected_keys or expected.intersection(result.missing_keys):
        raise ValueError("Checkpoint is incompatible with the trainable architecture.")
    return model.to(device).eval(), checkpoint


def train(config_path, seed=None, output=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if seed is not None:
        config["seed"] = seed
    seed_everything(config["seed"])
    codes = load_codes(config["codes"])
    train_rows, valid_rows = read_records(config["train"]), read_records(config["validation"])
    if not train_rows or not valid_rows:
        raise ValueError("Training and validation splits must be nonempty.")
    train_ids, valid_ids = {str(r["id"]) for r in train_rows}, {str(r["id"]) for r in valid_rows}
    if train_ids & valid_ids:
        raise ValueError("Training and validation note identifiers overlap.")
    vocab = build_vocab(train_rows, config.get("min_word_frequency", 1), config.get("vocab_size", 100000))
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cpu" and config.get("cpu_threads"):
        torch.set_num_threads(config["cpu_threads"])
    model = DescFuse(config, codes, len(vocab)).to(device)
    if config.get("word_embeddings"):
        matrix = np.load(config["word_embeddings"])
        if matrix.shape != tuple(model.encoder.embedding.weight.shape):
            raise ValueError("Embedding matrix must follow the train-only vocabulary order and dimensions.")
        with torch.no_grad():
            model.encoder.embedding.weight.copy_(torch.as_tensor(matrix, device=device))
            model.encoder.embedding.weight[0].zero_()
    if config.get("freeze_generator", False):
        for parameter in model.generator.parameters():
            parameter.requires_grad_(False)
    grouped = {"lora": [], "interaction": [], "classifier": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "lora" if name.startswith("generator.llm.") else (
            "interaction" if name.startswith(("fusion.cross_", "fusion.ffn")) else "classifier")
        grouped[group].append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": values, "lr": config[f"{group}_lr"]} for group, values in grouped.items() if values
    ], weight_decay=config.get("weight_decay", 0.0))
    train_loader = make_loader(train_rows, vocab, codes, config, True)
    valid_loader = make_loader(valid_rows, vocab, codes, config)
    destination = Path(output or config["output"])
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "best.pt").exists() or (destination / "history.jsonl").exists():
        raise FileExistsError("Choose a fresh output directory to preserve the previous run.")
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (destination / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    def file_hash(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    manifest = {"versions": {package: version(package) for package in ("torch", "transformers", "peft")},
                "device": str(device), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "input_sha256": {key: file_hash(config[key]) for key in ("codes", "train", "validation")}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    best, stale = -1.0, 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        if config.get("freeze_generator", False):
            model.generator.eval()
        total, total_icd, total_desc = 0.0, 0.0, 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            indices, weights = sample_candidates(batch["labels"], config["candidate_budget"])
            description_indices = indices if config["description_weight"] else None
            optimizer.zero_grad(set_to_none=True)
            logits, desc = model(batch, indices, description_indices, config.get("stop_classification_gradient", False))
            icd = sampled_bce(logits, batch["labels"], indices, weights)
            loss = icd + config["description_weight"] * desc
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
            optimizer.step()
            total += float(loss.detach())
            total_icd += float(icd.detach())
            total_desc += float(desc.detach())
        target, probability, _ = score_loader(model, valid_loader, device, config["inference_block_size"])
        threshold = select_threshold(target, probability)
        metrics = evaluate_metrics(target, probability, threshold, config.get("precision_k", 5))
        metrics.update(epoch=epoch, train_loss=total / len(train_loader),
                       classification_loss=total_icd / len(train_loader), description_loss=total_desc / len(train_loader))
        with (destination / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if metrics["micro_f1"] > best:
            best, stale = metrics["micro_f1"], 0
            save_checkpoint(destination / "best.pt", model, config, codes, vocab, threshold, epoch)
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    return destination / "best.pt"
