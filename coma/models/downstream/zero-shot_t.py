"""
Tuned zero-shot evaluation for CoMA downstream classification.

Improvements over zero-shot.py (especially for CheXpert):
  - Hybrid prompt bank: structured radiology templates + simple clinical sentences
  - Richer negative prompts for binary contrastive scoring
  - Optional max-similarity aggregation over prompt variants (CLIP-style)
  - Per-class AUROC, threshold tuning on a held-out split, confusion matrices

RSNA / COVIDx keep the original prompt sets and mean_prototype scoring by default.
CheXpert uses hybrid prompts with balanced top-k aggregation and expanded negatives.
"""

import itertools
import json
import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from sklearn.metrics import confusion_matrix
from torchmetrics import AUROC, Accuracy
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from coma.constants import CHEXPERT_CLASS_PROMPTS, CHEXPERT_COMPETITION_TASKS
from coma.models.downstream.zs_encoders import (
    add_zero_shot_model_args,
    build_zero_shot_datamodule,
    load_zero_shot_encoder,
)

RSNA_CLASS_PROMPTS = {
    "positive": [
        "Patient has pneumonia.",
        "Findings are consistent with pneumonia.",
        "There is airspace opacity suggesting pneumonia.",
    ],
    "negative": [
        "No evidence of pneumonia.",
        "The lungs are clear without pneumonia.",
        "No focal consolidation to suggest pneumonia.",
    ],
}

COVIDX_CLASS_PROMPTS = {
    0: [
        "This is a normal chest x-ray.",
        "No finding. Lungs are clear.",
        "No pneumonia or covid-19 infection.",
    ],
    1: [
        "This chest x-ray shows bacterial pneumonia.",
        "Findings consistent with pneumonia.",
        "There is pulmonary opacity suggesting pneumonia.",
    ],
    2: [
        "This chest x-ray shows covid-19 pneumonia.",
        "Findings consistent with covid-19 infection.",
        "There is ground-glass opacity suggesting covid-19.",
    ],
}

CHEXPERT_SIMPLE_POSITIVE_TEMPLATES = [
    "The chest X-ray shows {disease}.",
    "Findings are consistent with {disease}.",
    "There is evidence of {disease}.",
    "Radiographic findings suggest {disease}.",
]

CHEXPERT_NEGATIVE_TEMPLATES = [
    "No {disease}.",
    "No evidence of {disease}.",
    "There is no evidence of {disease}.",
    "The chest X-ray shows no {disease}.",
    "{disease} is not present.",
    "Lungs are clear without {disease}.",
    "The lungs are clear without {disease}.",
    "No radiographic evidence of {disease}.",
    "Chest x-ray is unremarkable for {disease}.",
    "No acute finding of {disease}.",
    "No sign of {disease} on this chest x-ray.",
    "There is no radiographic {disease}.",
]


def build_chexpert_structured_prompts(
    class_prompts: Dict = CHEXPERT_CLASS_PROMPTS,
    class_names: Sequence[str] = CHEXPERT_COMPETITION_TASKS,
) -> Dict[str, List[str]]:
    """Expand structured CheXpert templates into radiology-style sentences."""
    prompts: Dict[str, List[str]] = {}
    for class_name in class_names:
        templates = class_prompts[class_name]
        sentences = []
        for severity, subtype, location in itertools.product(
            templates["severity"], templates["subtype"], templates["location"]
        ):
            parts = [p.strip() for p in (severity, subtype, location) if p.strip()]
            if parts:
                sentences.append(" ".join(parts))
        if not sentences:
            sentences.append(class_name)
        prompts[class_name] = sentences
    return prompts


def build_chexpert_hybrid_prompts(
    class_names: Sequence[str] = CHEXPERT_COMPETITION_TASKS,
) -> Dict[str, List[str]]:
    """Structured prompts plus short clinical templates (deduplicated)."""
    prompts = build_chexpert_structured_prompts(class_names=class_names)
    for class_name in class_names:
        disease = class_name.lower()
        simple = [
            template.format(disease=disease)
            for template in CHEXPERT_SIMPLE_POSITIVE_TEMPLATES
        ]
        merged = simple + prompts[class_name]
        prompts[class_name] = list(dict.fromkeys(merged))
    return prompts


def build_chexpert_negative_prompts(
    class_names: Sequence[str] = CHEXPERT_COMPETITION_TASKS,
) -> Dict[str, List[str]]:
    prompts = {}
    for class_name in class_names:
        disease = class_name.lower()
        prompts[class_name] = [
            template.format(disease=disease)
            for template in CHEXPERT_NEGATIVE_TEMPLATES
        ]
    return prompts


def _tokenize_texts(text_encoder, texts: Sequence[str], device: torch.device):
    tokens = text_encoder.tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=112,
    )
    ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(ids)
    return ids, attention_mask, token_type_ids


@torch.no_grad()
def encode_texts(encoder, texts: Sequence[str], device: torch.device = None) -> torch.Tensor:
    """Encode texts into normalized global embeddings (N, D)."""
    return encoder.encode_texts(texts)


@torch.no_grad()
def encode_text_prompt_bank(
    encoder,
    prompt_bank: Sequence[str],
    device: torch.device = None,
) -> torch.Tensor:
    """Average normalized embeddings over prompt variants, then re-normalize."""
    if len(prompt_bank) == 0:
        raise ValueError("prompt_bank must not be empty")
    embs = encode_texts(encoder, prompt_bank, device)
    mean_emb = embs.mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=0)


@torch.no_grad()
def encode_images(encoder, imgs: torch.Tensor) -> torch.Tensor:
    """Encode images into normalized global embeddings (B, D)."""
    return encoder.encode_images(imgs)


def _aggregate_class_score(
    img_emb: torch.Tensor,
    prompt_embs: torch.Tensor,
    mode: str,
    topk: int = 5,
    pool_temperature: float = 0.05,
) -> torch.Tensor:
    """Aggregate image-text similarities for one class prompt set."""
    if prompt_embs.dim() == 1:
        prompt_embs = prompt_embs.unsqueeze(0)

    sims = img_emb @ prompt_embs.t()
    if mode == "mean_prototype":
        prototype = F.normalize(prompt_embs.mean(dim=0), p=2, dim=0)
        return img_emb @ prototype
    if mode == "max_similarity":
        return sims.max(dim=-1).values
    if mode == "topk_mean":
        k = min(topk, sims.size(-1))
        topk_vals, _ = sims.topk(k, dim=-1)
        return topk_vals.mean(dim=-1)
    if mode == "logsumexp":
        # De-bias for different bank sizes: subtract log(N).
        pooled = pool_temperature * torch.logsumexp(
            sims / pool_temperature, dim=-1
        )
        return pooled - torch.log(
            torch.tensor(sims.size(-1), dtype=sims.dtype, device=sims.device)
        )
    raise ValueError(f"Unsupported score_aggregate: {mode}")


def _aggregate_score_vector(
    scores: torch.Tensor,
    mode: str,
    topk: int = 5,
    pool_temperature: float = 0.05,
) -> torch.Tensor:
    """Aggregate already-computed (B, N) prompt scores."""
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    if mode == "mean_prototype":
        return scores.mean(dim=-1)
    if mode == "max_similarity":
        return scores.max(dim=-1).values
    if mode == "topk_mean":
        k = min(topk, scores.size(-1))
        topk_vals, _ = scores.topk(k, dim=-1)
        return topk_vals.mean(dim=-1)
    if mode == "logsumexp":
        pooled = pool_temperature * torch.logsumexp(scores / pool_temperature, dim=-1)
        return pooled - torch.log(
            torch.tensor(scores.size(-1), dtype=scores.dtype, device=scores.device)
        )
    raise ValueError(f"Unsupported score_aggregate: {mode}")


def _contrastive_native_logit(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    mode: str,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    pos_score = _aggregate_score_vector(pos_scores, mode, topk=topk)
    neg_score = _aggregate_score_vector(neg_scores, mode, topk=topk)
    return (pos_score - neg_score) / temperature


def _contrastive_logit(
    img_emb: torch.Tensor,
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    mode: str,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    pos_score = _aggregate_class_score(img_emb, pos_embs, mode, topk=topk)
    neg_score = _aggregate_class_score(img_emb, neg_embs, mode, topk=topk)
    return (pos_score - neg_score) / temperature


def _default_prompt_mode(dataset: str) -> str:
    if dataset == "chexpert":
        return "hybrid"
    return "structured"


def _default_score_aggregate(dataset: str) -> str:
    if dataset == "chexpert":
        return "ensemble"
    return "mean_prototype"


def _default_topk(dataset: str) -> int:
    if dataset == "chexpert":
        return 5
    return 3


def _default_tune_thresholds_on(dataset: str, split: str) -> Optional[str]:
    if dataset == "chexpert" and split == "test":
        return "valid"
    return None


class CoMATZeroShotClassifier:
    """Tuned zero-shot classifier with flexible prompt aggregation."""

    def __init__(
        self,
        encoder,
        dataset: str,
        temperature: Optional[float] = None,
        use_contrastive: bool = True,
        prompt_mode: Optional[str] = None,
        score_aggregate: Optional[str] = None,
        topk: Optional[int] = None,
    ):
        self.encoder = encoder
        self.coma = encoder
        self.dataset = dataset
        self.device = next(encoder.parameters()).device
        self.temperature = (
            temperature if temperature is not None else encoder.temperature
        )
        self.use_contrastive = use_contrastive
        self.prompt_mode = prompt_mode or _default_prompt_mode(dataset)
        self.score_aggregate = score_aggregate or _default_score_aggregate(dataset)
        self.topk = topk if topk is not None else _default_topk(dataset)
        self.multilabel = dataset in ("chexpert", "rsna")
        self.class_names = self._build_class_names()
        self.pos_prompt_texts, self.neg_prompt_texts = self._build_prompt_texts()
        self.pos_prompt_embs, self.neg_prompt_embs = self._build_prompt_embeddings()

    def _build_class_names(self) -> List[str]:
        if self.dataset == "chexpert":
            return list(CHEXPERT_COMPETITION_TASKS)
        if self.dataset == "rsna":
            return ["pneumonia"]
        if self.dataset == "covidx":
            return ["normal", "pneumonia", "covid-19"]
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _encode_prompt_list(self, prompt_bank: Sequence[str]) -> torch.Tensor:
        return encode_texts(self.encoder, prompt_bank, self.device)

    def _build_prompt_texts(
        self,
    ) -> Tuple[List[List[str]], Optional[List[Optional[List[str]]]]]:
        if self.dataset == "chexpert":
            if self.prompt_mode == "structured":
                prompt_dict = build_chexpert_structured_prompts()
            elif self.prompt_mode == "hybrid":
                prompt_dict = build_chexpert_hybrid_prompts()
            else:
                raise ValueError(f"Unsupported prompt_mode: {self.prompt_mode}")
            pos_texts = [prompt_dict[name] for name in self.class_names]
            if self.use_contrastive:
                neg_prompts = build_chexpert_negative_prompts(self.class_names)
                neg_texts = [neg_prompts[name] for name in self.class_names]
            else:
                neg_texts = None
            return pos_texts, neg_texts

        if self.dataset == "rsna":
            return [RSNA_CLASS_PROMPTS["positive"]], [RSNA_CLASS_PROMPTS["negative"]]

        if self.dataset == "covidx":
            pos_texts = [COVIDX_CLASS_PROMPTS[idx] for idx in range(len(self.class_names))]
            return pos_texts, None

        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _build_prompt_embeddings(
        self,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[Optional[torch.Tensor]]]]:
        if self.encoder.score_head in ("native", "query"):
            return None, None
        pos_embs = [self._encode_prompt_list(bank) for bank in self.pos_prompt_texts]
        if self.neg_prompt_texts is None:
            return pos_embs, None
        neg_embs = [
            self._encode_prompt_list(bank) if bank is not None else None
            for bank in self.neg_prompt_texts
        ]
        return pos_embs, neg_embs

    def _native_class_logit(
        self,
        imgs: torch.Tensor,
        pos_texts: Sequence[str],
        neg_texts: Optional[Sequence[str]],
    ) -> torch.Tensor:
        pos_scores = self.encoder.score_texts(imgs, pos_texts)
        if neg_texts is not None:
            neg_scores = self.encoder.score_texts(imgs, neg_texts)
            if self.score_aggregate == "ensemble":
                logit_mean = _contrastive_native_logit(
                    pos_scores, neg_scores, "mean_prototype", self.temperature, self.topk
                )
                logit_topk = _contrastive_native_logit(
                    pos_scores, neg_scores, "topk_mean", self.temperature, self.topk
                )
                return 0.5 * (logit_mean + logit_topk)
            return _contrastive_native_logit(
                pos_scores, neg_scores, self.score_aggregate, self.temperature, self.topk
            )
        agg_mode = (
            "mean_prototype" if self.score_aggregate == "ensemble" else self.score_aggregate
        )
        return _aggregate_score_vector(pos_scores, agg_mode, topk=self.topk) / self.temperature

    @torch.no_grad()
    def predict_logits(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.encoder.score_head in ("native", "query"):
            neg_lists = self.neg_prompt_texts or [None] * len(self.pos_prompt_texts)
            logits = [
                self._native_class_logit(imgs, pos_texts, neg_texts)
                for pos_texts, neg_texts in zip(self.pos_prompt_texts, neg_lists)
            ]
            return torch.stack(logits, dim=-1)

        img_emb = encode_images(self.encoder, imgs)
        logits = []
        neg_lists = self.neg_prompt_embs or [None] * len(self.pos_prompt_embs)

        for pos_embs, neg_embs in zip(self.pos_prompt_embs, neg_lists):
            if neg_embs is not None:
                if self.score_aggregate == "ensemble":
                    logit_mean = _contrastive_logit(
                        img_emb,
                        pos_embs,
                        neg_embs,
                        "mean_prototype",
                        self.temperature,
                        self.topk,
                    )
                    logit_topk = _contrastive_logit(
                        img_emb,
                        pos_embs,
                        neg_embs,
                        "topk_mean",
                        self.temperature,
                        self.topk,
                    )
                    logits.append(0.5 * (logit_mean + logit_topk))
                else:
                    logits.append(
                        _contrastive_logit(
                            img_emb,
                            pos_embs,
                            neg_embs,
                            self.score_aggregate,
                            self.temperature,
                            self.topk,
                        )
                    )
            else:
                agg_mode = (
                    "mean_prototype"
                    if self.score_aggregate == "ensemble"
                    else self.score_aggregate
                )
                pos_score = _aggregate_class_score(
                    img_emb, pos_embs, agg_mode, topk=self.topk
                )
                logits.append(pos_score / self.temperature)

        return torch.stack(logits, dim=-1)


@torch.no_grad()
def collect_predictions(
    classifier: CoMATZeroShotClassifier,
    dataloader,
    split: str = "test",
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_logits = []
    all_labels = []

    classifier.encoder.eval()
    for batch in tqdm(dataloader, desc=f"zero-shot-t/{split}"):
        imgs, labels = batch
        imgs = imgs.to(classifier.device)
        logits = classifier.predict_logits(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _binary_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    metric = AUROC(task="binary")
    probs = torch.sigmoid(logits)
    return metric(probs, labels.long()).item()


def _binary_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def _find_best_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_thresholds: int = 101,
) -> float:
    probs = torch.sigmoid(logits)
    labels = labels.long()
    thresholds = torch.linspace(0.0, 1.0, num_thresholds)
    best_threshold = 0.5
    best_j = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).long()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        tpr = tp / (tp + fn + 1e-8)
        tnr = tn / (tn + fp + 1e-8)
        j = tpr + tnr - 1.0
        if j > best_j:
            best_j = j
            best_threshold = threshold.item()

    return best_threshold


def _format_binary_confusion_matrix(
    cm: np.ndarray,
    class_name: str,
    threshold: float,
) -> str:
    tn, fp, fn, tp = cm.ravel()
    lines = [
        f"[{class_name}] threshold={threshold:.3f}",
        "                 Pred Neg    Pred Pos",
        f"True Neg         {tn:8d}    {fp:8d}",
        f"True Pos         {fn:8d}    {tp:8d}",
    ]
    return "\n".join(lines)


def _format_multiclass_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
) -> str:
    header = " " * 14 + "  ".join(f"{name:>12}" for name in class_names)
    rows = [header]
    for true_name, row in zip(class_names, cm):
        row_text = "  ".join(f"{value:12d}" for value in row)
        rows.append(f"True {true_name:<9} {row_text}")
    return "\n".join(rows)


def evaluate_zero_shot(
    classifier: CoMATZeroShotClassifier,
    dataloader,
    split: str = "test",
    thresholds: Optional[Union[float, Sequence[float]]] = None,
) -> Dict:
    logits, labels = collect_predictions(classifier, dataloader, split=split)
    results: Dict = {"split": split, "class_names": classifier.class_names}

    if classifier.multilabel:
        num_classes = logits.size(-1)
        probs = torch.sigmoid(logits)

        if num_classes == 1:
            results["metrics"] = {f"{split}_auc": _binary_auc(logits.squeeze(-1), labels.squeeze(-1))}
            label_vec = labels.squeeze(-1).numpy()
            prob_vec = probs.squeeze(-1).numpy()
            threshold = thresholds if isinstance(thresholds, float) else (
                thresholds[0] if thresholds is not None else 0.5
            )
            preds = (prob_vec >= threshold).astype(np.int64)
            cm = confusion_matrix(label_vec, preds, labels=[0, 1])
            results["thresholds"] = [threshold]
            results["confusion_matrices"] = {
                classifier.class_names[0]: cm.tolist(),
            }
            results["confusion_matrix_text"] = _format_binary_confusion_matrix(
                cm, classifier.class_names[0], threshold
            )
            return results

        per_class_auc = {}
        per_class_f1 = {}
        for idx, class_name in enumerate(classifier.class_names):
            per_class_auc[f"{split}_auc_{class_name}"] = _binary_auc(
                logits[:, idx], labels[:, idx]
            )

        metric = AUROC(
            task="multilabel",
            num_labels=num_classes,
            num_classes=num_classes,
        )
        macro_auc = metric(probs, labels.long()).item()
        results["metrics"] = {f"{split}_auc": macro_auc, **per_class_auc}

        if thresholds is None:
            class_thresholds = [0.5] * num_classes
        elif isinstance(thresholds, float):
            class_thresholds = [thresholds] * num_classes
        else:
            class_thresholds = list(thresholds)

        for idx, class_name in enumerate(classifier.class_names):
            label_vec = labels[:, idx].numpy()
            prob_vec = probs[:, idx].numpy()
            pred_vec = (prob_vec >= class_thresholds[idx]).astype(np.int64)
            per_class_f1[f"{split}_f1_{class_name}"] = _binary_f1(pred_vec, label_vec)

        results["metrics"][f"{split}_f1_macro"] = float(np.mean(list(per_class_f1.values())))
        results["metrics"].update(per_class_f1)

        confusion_matrices = {}
        confusion_texts = []
        for idx, class_name in enumerate(classifier.class_names):
            label_vec = labels[:, idx].numpy()
            pred_vec = (probs[:, idx].numpy() >= class_thresholds[idx]).astype(np.int64)
            cm = confusion_matrix(label_vec, pred_vec, labels=[0, 1])
            confusion_matrices[class_name] = cm.tolist()
            confusion_texts.append(
                _format_binary_confusion_matrix(cm, class_name, class_thresholds[idx])
            )

        results["thresholds"] = class_thresholds
        results["confusion_matrices"] = confusion_matrices
        results["confusion_matrix_text"] = "\n\n".join(confusion_texts)
        return results

    if labels.dim() > 1:
        labels = labels.squeeze(-1)

    metric = Accuracy(task="multiclass", num_classes=logits.size(-1), top_k=1)
    class_probs = F.softmax(logits, dim=-1)
    acc = metric(class_probs, labels.long())
    results["metrics"] = {f"{split}_acc": acc.item()}

    pred_labels = class_probs.argmax(dim=-1).numpy()
    true_labels = labels.long().numpy()
    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=list(range(len(classifier.class_names))),
    )
    results["confusion_matrices"] = {"multiclass": cm.tolist()}
    results["confusion_matrix_text"] = _format_multiclass_confusion_matrix(
        cm, classifier.class_names
    )
    return results


def tune_binary_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
) -> List[float]:
    thresholds = []
    for idx, _ in enumerate(class_names):
        thresholds.append(_find_best_threshold(logits[:, idx], labels[:, idx]))
    return thresholds


def get_dataloader(datamodule, split: str):
    if split == "train":
        return datamodule.train_dataloader()
    if split == "valid":
        return datamodule.val_dataloader()
    if split == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"Unsupported split: {split}")


def build_datamodule(args, encoder=None):
    if encoder is None:
        raise ValueError("encoder is required")
    return build_zero_shot_datamodule(args, encoder)


def print_results(results: Dict) -> None:
    print("=== Metrics ===")
    for name, value in results["metrics"].items():
        print(f"{name}: {value:.4f}")

    if "thresholds" in results:
        print("\n=== Thresholds ===")
        if results["class_names"] and len(results["class_names"]) == len(results["thresholds"]):
            for class_name, threshold in zip(results["class_names"], results["thresholds"]):
                print(f"{class_name}: {threshold:.4f}")
        else:
            print(", ".join(f"{threshold:.4f}" for threshold in results["thresholds"]))

    print("\n=== Confusion Matrix ===")
    print(results["confusion_matrix_text"])


def cli_main():
    parser = ArgumentParser(
        "Tuned zero-shot evaluation for CoMA / MedKLIP / KAD"
    )
    add_zero_shot_model_args(parser, coma_default_ckpt=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="chexpert",
        help="chexpert, rsna or covidx",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_pct", type=float, default=1.0)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Similarity temperature; defaults to checkpoint gia_temperature",
    )
    parser.add_argument(
        "--no_contrastive",
        action="store_true",
        help="Disable positive-negative contrastive scoring",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=None,
        choices=["structured", "hybrid"],
        help="CheXpert prompt bank; default hybrid for chexpert, structured otherwise",
    )
    parser.add_argument(
        "--score_aggregate",
        type=str,
        default=None,
        choices=["mean_prototype", "max_similarity", "topk_mean", "logsumexp", "ensemble"],
        help="Prompt aggregation; CheXpert default is ensemble (mean_prototype + topk_mean)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Top-k for topk_mean / ensemble; default 5 for chexpert, 3 otherwise",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary confusion matrices",
    )
    parser.add_argument(
        "--tune_thresholds_on",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="Explicitly tune per-class thresholds on this split",
    )
    parser.add_argument(
        "--no_tune_thresholds",
        action="store_true",
        help="Disable threshold tuning; use --threshold (default 0.5). "
        "Also disables CheXpert test auto-tuning on valid.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save metrics and confusion matrices as JSON",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_zero_shot_encoder(args, device)
    datamodule = build_datamodule(args, encoder)
    classifier = CoMATZeroShotClassifier(
        encoder=encoder,
        dataset=args.dataset,
        temperature=args.temperature,
        use_contrastive=not args.no_contrastive,
        prompt_mode=args.prompt_mode,
        score_aggregate=args.score_aggregate,
        topk=args.topk,
    )
    print(
        f"Using prompt_mode={classifier.prompt_mode}, "
        f"score_aggregate={classifier.score_aggregate}, topk={classifier.topk}"
    )

    thresholds: Optional[Union[float, Sequence[float]]] = args.threshold
    tune_split = None
    if not args.no_tune_thresholds:
        tune_split = args.tune_thresholds_on
        if tune_split is None:
            tune_split = _default_tune_thresholds_on(args.dataset, args.split)
    if tune_split is not None:
        tune_loader = get_dataloader(datamodule, tune_split)
        tune_logits, tune_labels = collect_predictions(
            classifier, tune_loader, split=tune_split
        )
        if classifier.multilabel:
            thresholds = tune_binary_thresholds(
                tune_logits, tune_labels, classifier.class_names
            )
            print(
                f"Tuned thresholds on {tune_split}: "
                + ", ".join(f"{value:.4f}" for value in thresholds)
            )
        else:
            print(
                "Threshold tuning is only used for binary/multi-label tasks; "
                "multiclass COVIDx uses argmax predictions."
            )

    eval_loader = get_dataloader(datamodule, args.split)
    results = evaluate_zero_shot(
        classifier,
        eval_loader,
        split=args.split,
        thresholds=thresholds,
    )
    print_results(results)

    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    cli_main()
