import torch
import torch.nn.functional as F


def sample_candidates(labels, budget):
    positive = labels.bool().any(0)
    pos_ids = positive.nonzero().flatten()
    negatives = (~positive).nonzero().flatten()
    if len(negatives) and budget <= len(pos_ids):
        count = 1
    else:
        count = min(len(negatives), max(0, budget - len(pos_ids)))
    sampled = negatives[torch.randperm(len(negatives), device=labels.device)[:count]]
    indices = torch.cat([pos_ids, sampled])
    weights = torch.ones(len(indices), device=labels.device)
    if count:
        weights[len(pos_ids):] = len(negatives) / count
    return indices, weights


def sampled_bce(logits, labels, indices, weights):
    # 以完整标签数归一化，校正均匀无放回负采样。
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels[:, indices], reduction="none")
    return (losses * weights[None]).sum() / labels.numel()
