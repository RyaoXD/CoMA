"""
Zero-shot classification for JBHI paper evaluation.

Follows the paper setting: encode class text prompts with the pretrained text
encoder, score images via cosine similarity in the GIA space, and predict
without any downstream training.

Binary tasks (CheXpert, RSNA):
    positive prompt: "The chest X-ray shows [disease]."
    negative prompt: "There is no evidence of [disease]."
    disease score = sim(image, positive) - sim(image, negative)

Multi-class task (COVIDx):
    assign each image to the class with the highest image-text similarity.

Metrics: AUROC for CheXpert/RSNA, ACC for COVIDx.
Designed for CoMA checkpoints with ResNet-50 image encoder (img_encoder=resnet_50).
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from argparse import ArgumentParser
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torchmetrics import AUROC, Accuracy
from tqdm import tqdm

from coma.constants import CHEXPERT_COMPETITION_TASKS
from coma.models.downstream.zs_encoders import (
    add_zero_shot_model_args,
    build_zero_shot_datamodule,
    load_zero_shot_encoder,
)

BINARY_POSITIVE_TEMPLATE = "The chest X-ray shows {disease}."
BINARY_NEGATIVE_TEMPLATE = "There is no evidence of {disease}."

COVIDX_CLASS_PROMPTS = {
    0: "The chest X-ray shows no finding.",
    1: "The chest X-ray shows pneumonia.",
    2: "The chest X-ray shows covid-19.",
}


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
def encode_images(encoder, imgs: torch.Tensor) -> torch.Tensor:
    """Encode images into normalized global embeddings (B, D)."""
    return encoder.encode_images(imgs)


def build_binary_prompts(diseases: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Build positive/negative prompt lists for binary zero-shot scoring."""
    positive = [
        BINARY_POSITIVE_TEMPLATE.format(disease=disease.lower())
        for disease in diseases
    ]
    negative = [
        BINARY_NEGATIVE_TEMPLATE.format(disease=disease.lower())
        for disease in diseases
    ]
    return positive, negative


class JBHIZeroShotClassifier:
    """Paper-aligned zero-shot classifier using GIA cosine similarity."""

    def __init__(self, encoder, dataset: str):
        self.encoder = encoder
        self.coma = encoder
        self.dataset = dataset
        self.device = next(encoder.parameters()).device
        self.multilabel = dataset in ("chexpert", "rsna")
        self.class_names = self._build_class_names()
        self.pos_texts, self.neg_texts = self._build_prompt_texts()
        self.pos_prototypes, self.neg_prototypes = self._build_prototypes()

    def _build_class_names(self) -> List[str]:
        if self.dataset == "chexpert":
            return list(CHEXPERT_COMPETITION_TASKS)
        if self.dataset == "rsna":
            return ["pneumonia"]
        if self.dataset == "covidx":
            return ["normal", "pneumonia", "covid-19"]
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _build_prompt_texts(self) -> Tuple[List[str], Optional[List[str]]]:
        if self.dataset in ("chexpert", "rsna"):
            return build_binary_prompts(self.class_names)
        if self.dataset == "covidx":
            return [COVIDX_CLASS_PROMPTS[idx] for idx in range(3)], None
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _build_prototypes(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.encoder.score_head in ("native", "query"):
            return None, None
        pos = encode_texts(self.encoder, self.pos_texts, self.device)
        neg = None
        if self.neg_texts is not None:
            neg = encode_texts(self.encoder, self.neg_texts, self.device)
        return pos, neg

    @torch.no_grad()
    def predict_logits(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.encoder.score_head in ("native", "query"):
            pos_sim = self.encoder.score_texts(imgs, self.pos_texts)
            if self.neg_texts is not None:
                neg_sim = self.encoder.score_texts(imgs, self.neg_texts)
                return pos_sim - neg_sim
            return pos_sim

        img_emb = encode_images(self.encoder, imgs)
        pos_sim = img_emb @ self.pos_prototypes.t()
        if self.neg_prototypes is not None:
            neg_sim = img_emb @ self.neg_prototypes.t()
            return pos_sim - neg_sim
        return pos_sim


@torch.no_grad()
def evaluate_zero_shot(
    classifier: JBHIZeroShotClassifier,
    dataloader,
    split: str = "test",
) -> Dict[str, float]:
    """Run zero-shot evaluation and return dataset-specific metrics."""
    all_logits = []
    all_labels = []

    classifier.encoder.eval()
    for batch in tqdm(dataloader, desc=f"zero-shot-jbhi/{classifier.dataset}/{split}"):
        imgs, labels = batch
        imgs = imgs.to(classifier.device)
        logits = classifier.predict_logits(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if classifier.multilabel:
        num_classes = logits.size(-1)
        if num_classes == 1:
            metric = AUROC(task="binary")
            probs = torch.sigmoid(logits.squeeze(-1))
            score = metric(probs, labels.squeeze(-1).long())
            return {f"{split}_auc": score.item()}
        metric = AUROC(
            task="multilabel",
            num_labels=num_classes,
            num_classes=num_classes,
        )
        probs = torch.sigmoid(logits)
        score = metric(probs, labels.long())
        return {f"{split}_auc": score.item()}

    metric = Accuracy(task="multiclass", num_classes=logits.size(-1), top_k=1)
    probs = F.softmax(logits, dim=-1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    score = metric(probs, labels.long())
    return {f"{split}_acc": score.item()}


def build_datamodule(args, encoder=None):
    if encoder is None:
        raise ValueError("encoder is required")
    return build_zero_shot_datamodule(args, encoder)


def load_coma_checkpoint(ckpt_path: str, img_encoder: str):
    """Load a pretrained CoMA checkpoint and verify the image encoder backbone."""
    from coma.models.coma.module import CoMA

    coma = CoMA.load_from_checkpoint(ckpt_path, strict=False)
    checkpoint_encoder = getattr(coma.hparams, "img_encoder", None)
    if checkpoint_encoder != img_encoder:
        raise ValueError(
            f"Checkpoint img_encoder={checkpoint_encoder!r}, "
            f"expected {img_encoder!r}. "
            "Use a ResNet-50 pretrained CoMA checkpoint for JBHI evaluation."
        )
    return coma


def run_single_dataset(args, encoder) -> Dict[str, float]:
    datamodule = build_datamodule(args, encoder)
    classifier = JBHIZeroShotClassifier(encoder=encoder, dataset=args.dataset)

    if args.split == "train":
        dataloader = datamodule.train_dataloader()
    elif args.split == "valid":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    return evaluate_zero_shot(classifier, dataloader, split=args.split)


def cli_main():
    parser = ArgumentParser(
        "JBHI zero-shot evaluation for CoMA / MedKLIP / KAD (ResNet-50)"
    )
    add_zero_shot_model_args(parser, coma_default_ckpt=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="chexpert",
        help="chexpert, rsna or covidx",
    )
    parser.add_argument(
        "--evaluate_all",
        action="store_true",
        default=False,
        help="Evaluate CheXpert, RSNA and COVIDx sequentially",
    )
    parser.add_argument(
        "--img_encoder",
        type=str,
        default="resnet_50",
        help="Expected image encoder in a CoMA checkpoint",
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
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_zero_shot_encoder(args, device)
    if args.model == "coma":
        checkpoint_encoder = getattr(encoder.coma.hparams, "img_encoder", None)
        if checkpoint_encoder != args.img_encoder:
            raise ValueError(
                f"Checkpoint img_encoder={checkpoint_encoder!r}, "
                f"expected {args.img_encoder!r}. "
                "Use a ResNet-50 pretrained CoMA checkpoint for JBHI evaluation."
            )

    datasets = ["chexpert", "rsna", "covidx"] if args.evaluate_all else [args.dataset]
    all_metrics = {}

    for dataset in datasets:
        args.dataset = dataset
        metrics = run_single_dataset(args, encoder)
        all_metrics.update({f"{dataset}/{name}": value for name, value in metrics.items()})
        for name, value in metrics.items():
            print(f"{dataset}/{name}: {value:.4f}")


if __name__ == "__main__":
    cli_main()
