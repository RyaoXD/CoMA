"""
Zero-shot downstream evaluation for CoMA.

Uses the pretrained global image-text alignment (GIA) space: encode images and
class text prompts, then predict via cosine similarity without any fine-tuning.
"""

import itertools
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from argparse import ArgumentParser
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torchmetrics import AUROC, Accuracy
from tqdm import tqdm

from coma.constants import CHEXPERT_CLASS_PROMPTS, CHEXPERT_COMPETITION_TASKS
from coma.models.downstream.zs_encoders import (
    add_zero_shot_model_args,
    build_zero_shot_datamodule,
    load_zero_shot_encoder,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


def build_chexpert_prompts(
    class_prompts: Dict = CHEXPERT_CLASS_PROMPTS,
    class_names: Sequence[str] = CHEXPERT_COMPETITION_TASKS,
) -> Dict[str, List[str]]:
    """Expand structured CheXpert templates into radiology-style sentences."""
    prompts = {}
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
    embs = encoder.encode_texts(prompt_bank)
    mean_emb = embs.mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=0)


@torch.no_grad()
def encode_images(encoder, imgs: torch.Tensor) -> torch.Tensor:
    """Encode images into normalized global embeddings (B, D)."""
    return encoder.encode_images(imgs)


class CoMAZeroShotClassifier:
    """Build class text prototypes and score images via GIA cosine similarity."""

    def __init__(
        self,
        encoder,
        dataset: str,
        temperature: Optional[float] = None,
        use_contrastive: bool = True,
    ):
        self.encoder = encoder
        self.coma = encoder
        self.dataset = dataset
        self.device = next(encoder.parameters()).device
        self.temperature = (
            temperature if temperature is not None else encoder.temperature
        )
        self.use_contrastive = use_contrastive
        self.multilabel = dataset in ("chexpert", "rsna")
        self.class_names = self._build_class_names()
        self.pos_prompt_lists, self.neg_prompt_lists = self._build_prompt_lists()
        self.pos_prototypes, self.neg_prototypes = self._build_prototypes()

    def _build_class_names(self) -> List[str]:
        if self.dataset == "chexpert":
            return list(CHEXPERT_COMPETITION_TASKS)
        if self.dataset == "rsna":
            return ["pneumonia"]
        if self.dataset == "covidx":
            return ["normal", "pneumonia", "covid-19"]
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _build_prompt_lists(self) -> Tuple[List[List[str]], Optional[List[List[str]]]]:
        if self.dataset == "chexpert":
            prompt_dict = build_chexpert_prompts()
            pos = [prompt_dict[name] for name in self.class_names]
            neg = None
            if self.use_contrastive:
                neg = [
                    [f"No {name.lower()}.", f"No evidence of {name.lower()}."]
                    for name in self.class_names
                ]
            return pos, neg

        if self.dataset == "rsna":
            return [RSNA_CLASS_PROMPTS["positive"]], [RSNA_CLASS_PROMPTS["negative"]]

        if self.dataset == "covidx":
            pos = [COVIDX_CLASS_PROMPTS[idx] for idx in range(len(self.class_names))]
            return pos, None

        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _build_prototypes(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.encoder.score_head != "clip":
            return None, None
        pos = torch.stack(
            [
                encode_text_prompt_bank(self.encoder, bank, self.device)
                for bank in self.pos_prompt_lists
            ],
            dim=0,
        )
        neg = None
        if self.neg_prompt_lists is not None:
            neg = torch.stack(
                [
                    encode_text_prompt_bank(self.encoder, bank, self.device)
                    for bank in self.neg_prompt_lists
                ],
                dim=0,
            )
        return pos, neg

    def _mean_native_scores(self, imgs: torch.Tensor, banks: List[List[str]]) -> torch.Tensor:
        logits = []
        for bank in banks:
            logits.append(self.encoder.score_texts(imgs, bank).mean(dim=-1))
        return torch.stack(logits, dim=-1)

    @torch.no_grad()
    def predict_logits(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.encoder.score_head == "native":
            pos_sim = self._mean_native_scores(imgs, self.pos_prompt_lists)
            if self.neg_prompt_lists is not None:
                neg_sim = self._mean_native_scores(imgs, self.neg_prompt_lists)
                return (pos_sim - neg_sim) / self.temperature
            return pos_sim / self.temperature

        img_emb = encode_images(self.encoder, imgs)
        pos_sim = img_emb @ self.pos_prototypes.t()
        if self.neg_prototypes is not None:
            neg_sim = img_emb @ self.neg_prototypes.t()
            logits = (pos_sim - neg_sim) / self.temperature
        else:
            logits = pos_sim / self.temperature
        return logits


@torch.no_grad()
def evaluate_zero_shot(
    classifier: CoMAZeroShotClassifier,
    dataloader,
    split: str = "test",
) -> Dict[str, float]:
    """Run zero-shot evaluation on a dataloader and return metrics."""
    all_logits = []
    all_labels = []

    classifier.encoder.eval()
    for batch in tqdm(dataloader, desc=f"zero-shot/{split}"):
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
            metric = AUROC(task="binary", num_classes=1)
            probs = torch.sigmoid(logits.squeeze(-1))
            score = metric(probs, labels.squeeze(-1).long())
            return {f"{split}_auc": score.item()}
        metric = AUROC(task="multilabel", num_labels=num_classes, num_classes=num_classes)
        probs = torch.sigmoid(logits)
        score = metric(probs, labels.long())
        return {f"{split}_auc": score.item()}

    metric = Accuracy(task="multiclass", num_classes=logits.size(-1), top_k=1)
    preds = F.softmax(logits, dim=-1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    score = metric(preds, labels.long())
    return {f"{split}_acc": score.item()}


def build_datamodule(args, encoder=None):
    if encoder is not None:
        return build_zero_shot_datamodule(args, encoder)
    raise ValueError("encoder is required")


def cli_main():
    parser = ArgumentParser("Zero-shot evaluation for CoMA / MedKLIP / KAD")
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
        help="Disable positive-negative contrastive scoring for CheXpert",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_zero_shot_encoder(args, device)
    datamodule = build_datamodule(args, encoder)

    classifier = CoMAZeroShotClassifier(
        encoder=encoder,
        dataset=args.dataset,
        temperature=args.temperature,
        use_contrastive=not args.no_contrastive,
    )

    if args.split == "train":
        dataloader = datamodule.train_dataloader()
    elif args.split == "valid":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    metrics = evaluate_zero_shot(classifier, dataloader, split=args.split)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    cli_main()
