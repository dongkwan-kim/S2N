from typing import List, Dict, Any

from sklearn.metrics import f1_score, accuracy_score
import torch


def _f1_score(logits, ys, avg_type="micro", is_multi_labels=False):
    preds = (logits > 0.0) if is_multi_labels else torch.argmax(logits, dim=-1)
    return f1_score(ys.cpu().detach(), preds.cpu().detach(), average=avg_type)


def _accuracy_score(logits, ys, is_multi_labels=False):
    preds = (logits > 0.0) if is_multi_labels else torch.argmax(logits, dim=-1)
    return accuracy_score(ys.cpu().detach(), preds.cpu().detach())


class Evaluator:

    def __init__(self, metrics: List[str], is_multi_labels):
        self.metrics = metrics
        self.is_multi_labels = is_multi_labels

    def __call__(self, logits, ys) -> Dict[str, Any]:
        evaluated = {}
        if "micro_f1" in self.metrics:
            evaluated["micro_f1"] = _f1_score(logits, ys, "micro", self.is_multi_labels)
        if "macro_f1" in self.metrics:
            evaluated["macro_f1"] = _f1_score(logits, ys, "macro", self.is_multi_labels)
        if "accuracy" in self.metrics:
            evaluated["accuracy"] = _accuracy_score(logits, ys, self.is_multi_labels)
        return evaluated
