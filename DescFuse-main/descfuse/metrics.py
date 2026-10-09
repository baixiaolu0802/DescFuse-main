import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def f1_scores(target, probability, threshold):
    predicted = probability >= threshold
    tp = (predicted & (target == 1)).sum(0)
    fp = (predicted & (target == 0)).sum(0)
    fn = (~predicted & (target == 1)).sum(0)
    denominator = 2 * tp + fp + fn
    macro = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0).mean()
    micro = 2 * tp.sum() / max(1, denominator.sum())
    return {"macro_f1": float(macro), "micro_f1": float(micro)}


def select_threshold(target, probability):
    candidates = np.linspace(0.01, 0.99, 99)
    scores = [f1_scores(target, probability, value)["micro_f1"] for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def equal_frequency_ece(target, confidence, bins=10):
    target, confidence = np.asarray(target).ravel(), np.asarray(confidence).ravel()
    if not len(target):
        return None
    groups = np.array_split(np.argsort(confidence, kind="stable"), bins)
    return float(sum(len(g) / len(target) * abs(target[g].mean() - confidence[g].mean())
                     for g in groups if len(g)))


def evaluate_metrics(target, probability, threshold, precision_k=5):
    result = f1_scores(target, probability, threshold)
    varying = (target.sum(0) > 0) & (target.sum(0) < len(target))
    result["macro_auc"] = float(roc_auc_score(target[:, varying], probability[:, varying], average="macro")) if varying.any() else None
    result["micro_auc"] = float(roc_auc_score(target.ravel(), probability.ravel())) if np.unique(target).size == 2 else None
    result["micro_pr_auc"] = float(average_precision_score(target.ravel(), probability.ravel())) if target.any() else None
    result["ece_equal_frequency"] = equal_frequency_ece(target, probability)
    k = min(precision_k, target.shape[1])
    top = np.argsort(-probability, axis=1, kind="stable")[:, :k]
    result[f"precision_at_{k}"] = float(np.take_along_axis(target, top, axis=1).mean())
    result["auc_eligible_codes"] = int(varying.sum())
    result["threshold"] = threshold
    return result
