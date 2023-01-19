import logging
import os
from pathlib import Path

import hydra 
import pytorch_lightning as pl
import torch
import wandb
from dotenv import find_dotenv, load_dotenv

# from google.cloud import secretmanager
from time import time

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.load_dataset import LoadImages
from src.models.model import DetrModel

from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="../conf", config_name="default_config.yaml")
def main(config: DictConfig):
    wandb.login(key='868e83ff4fd27d92c11d8aca0b8ed3af54078e19')
    wandb.init(project="project-mlops-object-detection")
    logger = logging.getLogger(__name__)
    logger.info("Training...")
    
    # client = secretmanager.SecretManagerServiceClient()
    # PROJECT_ID = "project-mlops-object-detection"

    # secret_id = "wandb-API"
    # resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    # response = client.access_secret_version(name=resource_name)
    # api_key = response.payload.data.decode("UTF-8")
    # os.environ["WANDB_API_KEY"] = api_key

    # wandb.init(project="project-mlops-object-detection", entity="mlops-object-detection", config=config)
    
    torch.manual_seed(config.train.seed)
    gpus = 0
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")
    
    loader = LoadImages(root_dir = HydraConfig.get().runtime.cwd)
    model = DetrModel(config)
    # saving the model
    output_model_dir = os.path.join(os.getcwd(), "model")
    os.makedirs(output_model_dir, exist_ok=True)
    # output_model_path = os.path.join(output_model_dir, "model.pt")

    model_checkpoint = ModelCheckpoint(dirpath='gs://od-model-checkpoints/')

    trainer = Trainer(
        max_epochs= config.train.epochs,
        gpus=gpus,
        logger=pl.loggers.WandbLogger(project="project-mlops-object-detection", log_model="all", config=config), # TODO
        # val_check_interval=1.0,
        # check_val_every_n_epoch=1,
        # gradient_clip_val=1.0,
        # default_root_dir='gs://od-model-checkpoints/'
        callbacks=[model_checkpoint]
    )
    trainer.fit(
        model,
        train_dataloaders=loader.get_dataloader(config.train.dataset, config.train.batch_size),
    )

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
