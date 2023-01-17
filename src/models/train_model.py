import logging
import os
from pathlib import Path

import hydra 
import pytorch_lightning as pl
import torch
# import wandb
# from dotenv import find_dotenv, load_dotenv
# from google.cloud import secretmanager
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.data.load_dataset import LoadImages
from src.models.model import DetrModel


@hydra.main(config_path="../conf", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Training...")
    torch.manual_seed(config.train.seed)
    gpus = 0
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")
    
    src_models_path = os.path.dirname(__file__)
    src_path = os.path.dirname(src_models_path)
    root_folder = os.path.dirname(src_path)
    print(root_folder)
    loader = LoadImages(paths = {
        #'voc': 'E:/mlops/data/raw/voc',
        'voc': os.path.join(root_folder,'data','raw','voc'),
        #'coco': 'E:/mlops/data/raw/coco/images/val2017/',
        'coco': os.path.join(root_folder,'data','raw','coco','images','val2017'),
        #'coco_annotations': 'E:/mlops/data/raw/coco/annotations/instances_val2017.json'
        'coco_annotations': os.path.join(root_folder,'data','raw','coco','annotations','instances_val2017.json')
        })
    model = DetrModel(config)

    trainer = Trainer(
        max_epochs= config.train.epochs,
        gpus=gpus,
        logger=pl.loggers.WandbLogger(project="project-mlops-object-detection", config=config), # TODO
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model,
        train_dataloaders=loader.get_dataloader(config.train.dataset, config.train.batch_size),
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()