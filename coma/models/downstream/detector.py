import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger

from coma.datasets.data_module import DataModule
from coma.datasets.detection_dataset import (OBJCXRDetectionDataset,
                                             RSNADetectionDataset)
from coma.datasets.transforms import DetectionDataTransforms
from coma.models.backbones.detector_backbone import ResNetDetector
from coma.models.backbones.transformer_seg import SETRModel
from coma.models.coma.module import CoMA
from coma.models.downstream.ssl_detector import SSLDetector
from coma.models.downstream.zs_encoders import (
    DEFAULT_KAD_CKPT,
    resolve_checkpoint_file,
    strip_module_prefix,
)

# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.backends.cudnn.allow_tf32 = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COMA_CKPT = None


def load_kad_resnet_weights(ckpt_path: str) -> dict:
    """Map official KAD image_encoder.resnet.* keys onto torchvision ResNet-50."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "image_encoder" not in ckpt:
        raise ValueError(
            f"KAD checkpoint should contain image_encoder, got keys="
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
    state = strip_module_prefix(ckpt["image_encoder"])
    mapped = {}
    for key, value in state.items():
        if not key.startswith("resnet."):
            continue
        new_key = key[len("resnet."):]
        if new_key.startswith("fc."):
            continue
        mapped[new_key] = value
    if not any(key.startswith("layer4.") for key in mapped):
        raise ValueError(
            "KAD checkpoint is missing resnet.layer4; use KAD-512 or KAD-1024."
        )
    return mapped


def cli_main():
    parser = ArgumentParser("Finetuning of object detection task for MGCA")
    parser.add_argument("--base_model", type=str, default="resnet_50", help="vit or resnet_50")
    parser.add_argument(
        "--model",
        type=str,
        default="coma",
        choices=["coma", "kad"],
        help="Pretrained visual backbone: CoMA lightning ckpt or official KAD",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Checkpoint path (required for CoMA). "
            "For --model kad, defaults to COMA_KAD_CKPT / --ckpt_path."
        ),
    )
    parser.add_argument("--dataset", type=str,
                        default="rsna", help="rsna or object_cxr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.model == "kad":
        args.ckpt_path = resolve_checkpoint_file(
            args.ckpt_path or DEFAULT_KAD_CKPT, "kad"
        )
        args.base_model = "resnet_50"
        if args.experiment_name == "test":
            args.experiment_name = "kad512"
    else:
        args.ckpt_path = args.ckpt_path or DEFAULT_COMA_CKPT
        if not args.ckpt_path:
            raise ValueError("Please pass --ckpt_path to a pretrained CoMA checkpoint.")

    # args.deterministic = True
    args.max_epochs = 50
    # args.accelerator = "gpu"
    args.gpus = 1

    seed_everything(args.seed)

    if args.dataset == "rsna":
        datamodule = DataModule(RSNADetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    elif args.dataset == "object_cxr":
        datamodule = DataModule(OBJCXRDetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    else:
        raise RuntimeError(f"{args.dataset} does not exist!")

    if args.base_model == "resnet_50":
        args.img_encoder = ResNetDetector("resnet_50", pretrained=args.model == "coma")
        if args.model == "kad":
            kad_state = load_kad_resnet_weights(args.ckpt_path)
            incompatible = args.img_encoder.model.load_state_dict(kad_state, strict=False)
            missing = [
                key for key in getattr(incompatible, "missing_keys", [])
                if not key.startswith("fc.")
            ]
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            print(
                f"Loaded KAD-512 ResNet from {args.ckpt_path} "
                f"(mapped={len(kad_state)}, missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if missing:
                raise RuntimeError(f"KAD ResNet missing keys: {missing[:12]}")
            args.img_encoder.prompt = None
        elif args.ckpt_path:
            ckpt = torch.load(args.ckpt_path)
            ckpt_dict = dict()
            for k, v in ckpt["state_dict"].items():
                if k.startswith("img_encoder_q.model"):
                    new_k = ".".join(k.split(".")[2:])
                    ckpt_dict[new_k] = v

            args.img_encoder.model.load_state_dict(ckpt_dict)
            args.img_encoder.prompt = getattr(ckpt, 'prompt_vector',
                                          getattr(ckpt, 'propompt_vector', None))
        else:
            args.img_encoder.prompt = None

        # Freeze encoder
        for param in args.img_encoder.parameters():
            param.requires_grad = False
    elif args.base_model == "vit":
        if args.model == "kad":
            raise ValueError("KAD detection uses ResNet-50; do not pass --base_model vit")
        coma = CoMA.load_from_checkpoint(args.ckpt_path)
        args.img_encoder = coma.img_encoder_q.model
        args.img_encoder.prompt = getattr(coma, 'prompt_vector',
                                          getattr(coma, 'propompt_vector', None))

    model = SSLDetector(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/detection/{args.experiment_name}/{args.dataset}_{args.data_pct}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_mAP", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_mAP", min_delta=0.,
                      patience=20, verbose=False, mode="max")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data/detection/{args.dataset}_{args.data_pct}/{args.experiment_name}")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="coma_jbhi_detection", save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{args.experiment_name}_{extension}")
    tensorboard_logger = TensorBoardLogger(logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger
    )
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
