"""Unified image-text encoders for CoMA / MedKLIP / KAD zero-shot evaluation.

The three downstream scripts score class prompts in a shared embedding space.
CoMA uses global cosine similarity. MedKLIP and KAD default to their official
query heads (existence classifier / TQN), and can fall back to cosine pooling
with ``--score_head cosine``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from coma.models.coma.module import CoMA


# Optional local defaults; prefer passing --ckpt_path / --bert_path / env vars.
MEDKLIP_SAMPLE_ROOT = Path(
    os.environ.get("COMA_MEDKLIP_ROOT", "")
).expanduser() if os.environ.get("COMA_MEDKLIP_ROOT") else None
DEFAULT_MEDKLIP_CKPT = os.environ.get("COMA_MEDKLIP_CKPT")
DEFAULT_KAD_CKPT = os.environ.get("COMA_KAD_CKPT")
DEFAULT_MEDKLIP_BERT = os.environ.get(
    "COMA_MEDKLIP_BERT", "emilyalsentzer/Bio_ClinicalBERT"
)
DEFAULT_KAD_BERT = os.environ.get("COMA_KAD_BERT")
DEFAULT_DISEASE_BOOK = (
    str(MEDKLIP_SAMPLE_ROOT / "observation explanation.json")
    if MEDKLIP_SAMPLE_ROOT is not None
    else None
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CKPT_CANDIDATES = (
    "checkpoint_final.pth",
    "checkpoint.pth",
    "checkpoint_best.pth",
    "checkpoint_21.pth",
    "best_valid.pt",
    "best.pth",
    "model.pth",
    "KAD/KAD_Models/KAD_512/best_valid.pt",
)


def add_external_model_args(
    parser: ArgumentParser, coma_default_ckpt: Optional[str] = None
) -> ArgumentParser:
    parser.add_argument(
        "--model",
        type=str,
        default="coma",
        choices=["coma", "medklip", "kad"],
        help="Zero-shot backbone: CoMA checkpoint, official MedKLIP, or official KAD",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Checkpoint path. Defaults to the CoMA ckpt of this script, "
            f"{DEFAULT_MEDKLIP_CKPT} for MedKLIP, or {DEFAULT_KAD_CKPT} for KAD"
        ),
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        default=None,
        help="Text backbone. MedKLIP: Bio_ClinicalBERT; KAD: Med-KEBERT tokenizer/config",
    )
    parser.add_argument(
        "--score_head",
        type=str,
        default="auto",
        choices=["auto", "cosine", "query", "clip", "native"],
        help="auto: CLIP cosine for CoMA, query/native head for MedKLIP and KAD",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Center-crop size. Default 224 for CoMA/MedKLIP, 512 for KAD",
    )
    parser.add_argument(
        "--disease_book",
        type=str,
        default=DEFAULT_DISEASE_BOOK,
        help="MedKLIP observation-explanation JSON used to initialize the query book",
    )
    parser.set_defaults(coma_default_ckpt=coma_default_ckpt)
    return parser


def add_zero_shot_model_args(
    parser: ArgumentParser, coma_default_ckpt: Optional[str] = None
) -> ArgumentParser:
    return add_external_model_args(parser, coma_default_ckpt)


def default_image_size(model_name: str) -> int:
    return 512 if model_name == "kad" else 224


def default_score_head(model_name: str, score_head: str) -> str:
    aliases = {"cosine": "clip", "query": "native"}
    score_head = aliases.get(score_head, score_head)
    if score_head != "auto":
        return score_head
    return "clip" if model_name == "coma" else "native"


def is_query_head(score_head: str) -> bool:
    return score_head in ("query", "native")


def resolve_ckpt_path(
    model_name: str, ckpt_path: Optional[str], coma_default: Optional[str]
) -> str:
    if model_name == "coma":
        path = ckpt_path or coma_default
    elif model_name == "medklip":
        path = ckpt_path or DEFAULT_MEDKLIP_CKPT
    elif model_name == "kad":
        path = ckpt_path or DEFAULT_KAD_CKPT
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    if not path:
        raise ValueError(
            f"Missing checkpoint for --model {model_name}. "
            "Pass --ckpt_path or set the corresponding COMA_*_CKPT env var."
        )
    return resolve_checkpoint_file(path, model_name)


def resolve_checkpoint_file(path: str, model_name: str) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        for name in _CKPT_CANDIDATES:
            hit = candidate / name
            if hit.is_file():
                return str(hit)
        hits = sorted(list(candidate.glob("*.pth")) + list(candidate.glob("*.pt")))
        if len(hits) == 1:
            return str(hits[0])
        if hits:
            names = ", ".join(p.name for p in hits)
            raise FileNotFoundError(
                f"{candidate} 里有多个权重文件 ({names})，请用 --ckpt_path 指定其中一个。"
            )
    hint = {
        "coma": "CoMA 的 .ckpt",
        "medklip": f"官方 MedKLIP 权重，放到 {DEFAULT_MEDKLIP_CKPT}",
        "kad": f"官方 KAD 权重，放到 {DEFAULT_KAD_CKPT}",
    }[model_name]
    raise FileNotFoundError(
        f"找不到 {model_name} checkpoint: {path}\n请把下载好的权重放到: {hint}"
    )


def resolve_bert_path(model_name: str, bert_path: Optional[str]) -> str:
    if bert_path:
        return bert_path

    def _usable(path: Optional[str]) -> bool:
        if not path:
            return False
        if not path.startswith("/"):
            return True
        candidate = Path(path)
        if not candidate.exists():
            return False
        if candidate.is_file():
            return True
        return any(
            (candidate / name).exists()
            for name in ("config.json", "vocab.txt", "tokenizer_config.json", "pytorch_model.bin")
        )

    if model_name == "medklip":
        if _usable(DEFAULT_MEDKLIP_BERT):
            return DEFAULT_MEDKLIP_BERT
        return "emilyalsentzer/Bio_ClinicalBERT"
    if model_name == "kad":
        for candidate in (
            DEFAULT_KAD_BERT,
            "xmcmic/Med-KEBERT",
            DEFAULT_MEDKLIP_BERT,
        ):
            if _usable(candidate):
                return candidate
        return "xmcmic/Med-KEBERT"
    raise ValueError(f"No default bert path for {model_name}")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(not key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def coma_to_imagenet(imgs: torch.Tensor) -> torch.Tensor:
    """Convert CoMA/MGCA normalization (mean=std=0.5) to ImageNet stats."""
    pixel = (imgs * 0.5 + 0.5).clamp(0.0, 1.0)
    mean = imgs.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = imgs.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pixel - mean) / std


def maybe_resize(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if imgs.shape[-1] == image_size and imgs.shape[-2] == image_size:
        return imgs
    return F.interpolate(
        imgs, size=(image_size, image_size), mode="bicubic", align_corners=False
    )


class ZeroShotEncoder(nn.Module):
    """Minimal interface consumed by the three zero-shot scripts."""

    model_name: str = "coma"
    score_head: str = "cosine"
    default_temperature: float = 0.07
    image_size: int = 224

    @property
    def temperature(self) -> float:
        return self.default_temperature

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_visual(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.encode_images(imgs)

    def score_visual(self, visual: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        if text_embs.dim() == 1:
            text_embs = text_embs.unsqueeze(0)
        return visual @ text_embs.t()

    def score(self, imgs: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        return self.score_visual(self.encode_visual(imgs), text_embs)

    def score_texts(self, imgs: torch.Tensor, texts: Sequence[str]) -> torch.Tensor:
        return self.score(imgs, self.encode_texts(texts))


class CoMAEncoder(ZeroShotEncoder):
    def __init__(self, coma: CoMA, score_head: str = "cosine"):
        super().__init__()
        if score_head not in ("cosine", "clip"):
            raise ValueError("CoMA zero-shot only supports --score_head cosine/clip")
        self.coma = coma
        self.model_name = "coma"
        self.score_head = "clip"
        self.image_size = 224
        self.default_temperature = float(getattr(coma.hparams, "gia_temperature", 0.07))

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        text_encoder = self.coma.text_encoder_q
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
        report_feat, _, _, _ = text_encoder(ids, attention_mask, token_type_ids)
        report_emb = text_encoder.global_embed(report_feat)
        return F.normalize(report_emb, p=2, dim=-1)

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        img_feat, _, _ = self.coma._encode_images(imgs)
        img_emb = self.coma.img_encoder_q.global_embed(img_feat)
        return F.normalize(img_emb, p=2, dim=-1)


def _import_official_medklip():
    pkg = "coma_ext_medklip"
    module_name = f"{pkg}.models.model_MedKLIP"
    if module_name in sys.modules:
        return sys.modules[module_name].MedKLIP

    root = MEDKLIP_SAMPLE_ROOT
    if not (root / "models" / "model_MedKLIP.py").is_file():
        raise FileNotFoundError(f"MedKLIP 源码不存在: {root}")

    pkg_mod = types.ModuleType(pkg)
    pkg_mod.__path__ = [str(root)]
    sys.modules[pkg] = pkg_mod

    models_mod = types.ModuleType(f"{pkg}.models")
    models_mod.__path__ = [str(root / "models")]
    sys.modules[f"{pkg}.models"] = models_mod

    transformer_spec = importlib.util.spec_from_file_location(
        f"{pkg}.models.transformer", root / "models" / "transformer.py"
    )
    transformer_mod = importlib.util.module_from_spec(transformer_spec)
    sys.modules[f"{pkg}.models.transformer"] = transformer_mod
    transformer_spec.loader.exec_module(transformer_mod)

    model_spec = importlib.util.spec_from_file_location(
        module_name, root / "models" / "model_MedKLIP.py"
    )
    model_mod = importlib.util.module_from_spec(model_spec)
    sys.modules[module_name] = model_mod
    model_spec.loader.exec_module(model_mod)
    return model_mod.MedKLIP


class MedKLIPEncoder(ZeroShotEncoder):
    def __init__(
        self,
        ckpt_path: str,
        bert_path: str,
        disease_book: str,
        score_head: str = "query",
        image_size: int = 224,
    ):
        super().__init__()
        self.model_name = "medklip"
        self.score_head = "native" if score_head in ("query", "native") else "clip"
        self.image_size = image_size
        self.default_temperature = 1.0 if self.score_head == "native" else 0.07
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path)

        if not Path(disease_book).is_file():
            raise FileNotFoundError(f"MedKLIP disease book 不存在: {disease_book}")
        book_json = json.load(open(disease_book, "r"))
        book_texts = [book_json[key] for key in book_json]
        book_tokens = self.tokenizer(
            book_texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        config = {
            "d_model": 256,
            "res_base_model": "resnet50",
            "H": 4,
            "N": 4,
            "dropout": 0.1,
            "attribute_set_size": 2,
            "text_encoder": bert_path,
            "num_queries": 75,
        }
        MedKLIP = _import_official_medklip()
        self.model = MedKLIP(config, book_tokens)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        incompatible = self.model.load_state_dict(strip_module_prefix(state), strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing or unexpected:
            print(
                f"[MedKLIP] load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
            )

    def _prepare_images(self, imgs: torch.Tensor) -> torch.Tensor:
        return maybe_resize(coma_to_imagenet(imgs), self.image_size)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        hidden = self.bert(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        )
        cls = hidden.last_hidden_state[:, 0, :]
        emb = self.model.disease_embedding_layer(cls)
        if self.score_head == "native":
            return emb
        return F.normalize(emb, p=2, dim=-1)

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        patches = self.model.image_encoder(self._prepare_images(imgs))
        pooled = patches.mean(dim=1)
        return F.normalize(pooled, p=2, dim=-1)

    def encode_visual(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.score_head == "clip":
            return self.encode_images(imgs)
        return self.model.image_encoder(self._prepare_images(imgs))

    def score_visual(self, visual: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        if text_embs.dim() == 1:
            text_embs = text_embs.unsqueeze(0)
        if self.score_head == "clip":
            return visual @ text_embs.t()
        batch = visual.shape[0]
        features = visual.transpose(0, 1)
        query = text_embs.unsqueeze(1).repeat(1, batch, 1)
        decoded, _ = self.model.decoder(
            query, features, memory_key_padding_mask=None, pos=None, query_pos=None
        )
        decoded = self.model.dropout_feas(decoded)
        logits = self.model.classifier(decoded).transpose(0, 1)
        return logits[..., 1] - logits[..., 0]


class KADEncoder(ZeroShotEncoder):
    def __init__(
        self,
        ckpt_path: str,
        bert_path: str,
        score_head: str = "query",
        image_size: int = 512,
    ):
        super().__init__()
        from coma.models.downstream.external.kad_models import (
            CLP_clinical,
            ModelRes,
            ModelRes512,
            TQN_Model,
        )

        self.model_name = "kad"
        self.score_head = "native" if score_head in ("query", "native") else "clip"
        self.image_size = image_size
        self.default_temperature = 1.0 if self.score_head == "native" else 0.07
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path, do_lower_case=True)
        self.text_encoder = CLP_clinical(bert_model_name=bert_path)
        if image_size <= 224:
            self.image_encoder = ModelRes(res_base_model="resnet50")
        else:
            self.image_encoder = ModelRes512(res_base_model="resnet50")
        self.tqn = TQN_Model()

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unexpected KAD checkpoint type: {type(checkpoint)}")
        if "image_encoder" not in checkpoint:
            for nested_key in ("state_dict", "model_state", "net"):
                nested = checkpoint.get(nested_key)
                if isinstance(nested, dict) and "image_encoder" in nested:
                    checkpoint = nested
                    break
        if "image_encoder" not in checkpoint or "text_encoder" not in checkpoint:
            raise ValueError(
                "KAD checkpoint 应包含 image_encoder / text_encoder / model 三个 state_dict。"
                f" 实际 keys={list(checkpoint.keys())}"
            )
        self.image_encoder.load_state_dict(strip_module_prefix(checkpoint["image_encoder"]))
        self.text_encoder.load_state_dict(strip_module_prefix(checkpoint["text_encoder"]))
        tqn_state = checkpoint.get("model", checkpoint.get("tqn"))
        if tqn_state is None:
            raise ValueError("KAD checkpoint 缺少 TQN 的 model 权重")
        self.tqn.load_state_dict(strip_module_prefix(tqn_state))

    def _prepare_images(self, imgs: torch.Tensor) -> torch.Tensor:
        return maybe_resize(coma_to_imagenet(imgs), self.image_size)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        emb = self.text_encoder.encode_text(tokens)
        if self.score_head == "native":
            return emb
        return F.normalize(emb, p=2, dim=-1)

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        _, pooled = self.image_encoder(self._prepare_images(imgs))
        return F.normalize(pooled, p=2, dim=-1)

    def encode_visual(self, imgs: torch.Tensor) -> torch.Tensor:
        patches, pooled = self.image_encoder(self._prepare_images(imgs))
        if self.score_head == "clip":
            return F.normalize(pooled, p=2, dim=-1)
        return patches

    def score_visual(self, visual: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        if text_embs.dim() == 1:
            text_embs = text_embs.unsqueeze(0)
        if self.score_head == "clip":
            return visual @ text_embs.t()
        logits = self.tqn(visual, text_embs)
        return logits[..., 1] - logits[..., 0]


def load_zero_shot_encoder(
    args: Namespace,
    device: torch.device,
    coma_default_ckpt: Optional[str] = None,
    expected_img_encoder: Optional[str] = None,
) -> ZeroShotEncoder:
    coma_default_ckpt = coma_default_ckpt or getattr(args, "coma_default_ckpt", None)
    if not coma_default_ckpt:
        raise ValueError("coma_default_ckpt is required")
    model_name = args.model
    score_head = default_score_head(model_name, getattr(args, "score_head", "auto"))
    image_size = getattr(args, "image_size", None) or default_image_size(model_name)
    args.image_size = image_size
    args.score_head = score_head
    ckpt_path = resolve_ckpt_path(model_name, getattr(args, "ckpt_path", None), coma_default_ckpt)
    args.ckpt_path = ckpt_path

    if model_name == "coma":
        coma = CoMA.load_from_checkpoint(ckpt_path, strict=False)
        if expected_img_encoder is not None:
            checkpoint_encoder = getattr(coma.hparams, "img_encoder", None)
            if checkpoint_encoder != expected_img_encoder:
                raise ValueError(
                    f"Checkpoint img_encoder={checkpoint_encoder!r}, "
                    f"expected {expected_img_encoder!r}. "
                    "Use a ResNet-50 pretrained CoMA checkpoint for JBHI evaluation."
                )
        encoder: ZeroShotEncoder = CoMAEncoder(coma, score_head=score_head)
    elif model_name == "medklip":
        bert_path = resolve_bert_path(model_name, getattr(args, "bert_path", None))
        encoder = MedKLIPEncoder(
            ckpt_path=ckpt_path,
            bert_path=bert_path,
            disease_book=getattr(args, "disease_book", None) or DEFAULT_DISEASE_BOOK,
            score_head=score_head,
            image_size=image_size,
        )
    elif model_name == "kad":
        bert_path = resolve_bert_path(model_name, getattr(args, "bert_path", None))
        encoder = KADEncoder(
            ckpt_path=ckpt_path,
            bert_path=bert_path,
            score_head=score_head,
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print(
        f"Loaded {model_name} from {ckpt_path} "
        f"(score_head={score_head}, image_size={image_size})"
    )
    return encoder


def build_zero_shot_datamodule(args: Namespace, encoder: Optional[ZeroShotEncoder] = None):
    from coma.datasets.classification_dataset import (
        CheXpertImageDataset,
        COVIDXImageDataset,
        RSNAImageDataset,
    )
    from coma.datasets.data_module import DataModule
    from coma.datasets.transforms import DataTransforms, Moco2Transform

    if encoder is not None:
        crop_size = int(getattr(args, "image_size", None) or encoder.image_size)
    else:
        crop_size = int(getattr(args, "image_size", None) or default_image_size(args.model))
    imsize = max(256, crop_size)
    args.image_size = crop_size
    common = dict(
        data_pct=args.data_pct,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=crop_size,
        imsize=imsize,
    )
    if args.dataset == "chexpert":
        return DataModule(CheXpertImageDataset, None, Moco2Transform, **common)
    if args.dataset == "rsna":
        return DataModule(RSNAImageDataset, None, DataTransforms, **common)
    if args.dataset == "covidx":
        return DataModule(COVIDXImageDataset, None, DataTransforms, **common)
    raise ValueError(f"Unsupported dataset: {args.dataset}")
