import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from coma.datasets.data_module import DataModule
from coma.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from coma.datasets.transforms import DataTransforms

from coma.models.coma.module import CoMA
from coma.paths import CKPT_DIR, LOG_DIR

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = CoMA.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50
    args.gpus = 1

    if args.experiment_name == "test":
        args.gpus = 1
        args.batch_size = 16
        args.img_encoder = "resnet_50"
        args.rewrite = True
        args.concept_decompose = True
        args.max_concepts = 30
        
        args.prompt_vector = False
        args.use_dpl = False
        
        args.freeze_prototypes_epochs = 5
        args.instance_prototype = "random"
        args.concept_prototype = "random"
        args.num_instance_prototypes = 500
        args.num_concept_prototypes = 500
        args.prototype_orth = True
        
        # args.rewrite = True
        # args.concept_decompose = True
        # args.max_concepts = 40
        
        # args.prompt_vector = True
        # args.use_dpl = True
        
        # args.vector_length = 40
        # args.freeze_prototypes_epochs = 5
        # args.instance_prototype = "random"
        # args.concept_prototype = "random"
        # args.num_instance_prototypes = 500
        # args.num_concept_prototypes = 500
        # args.prototype_orth = True
        
    
    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers,
                            args.rewrite, args.concept_decompose)

    # Add load from checkpoint
    model = CoMA(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = str(CKPT_DIR / "coma" / args.experiment_name / extension)
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val/loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=2),
        EarlyStopping(monitor="val/loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = str(LOG_DIR / "coma" / args.experiment_name)
    os.makedirs(logger_dir, exist_ok=True)
    logger = TensorBoardLogger(logger_dir, name=extension)
    # logger = WandbLogger(
    #     project="CoMA", save_dir=logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=logger,
        precision=args.mixed_precision,
        )

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule,
                ckpt_path=args.ckpt_path
                )

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    main()
