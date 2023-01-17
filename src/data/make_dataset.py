# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms

import hydra
from hydra.core.hydra_config import HydraConfig
from load_dataset import CocoDataset, VOCDataset


@hydra.main(config_path='../conf/dataset', config_name='coco_2017')
def main(cfg):
    """ 
    Downloads raw data to the data/raw/ folder. Default option downloads coco_2017 data.
    To download voc data run script with --config-name=voc
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    root_path = HydraConfig.get().runtime.cwd
    input_filepath = cfg.hyperparameters.input_filepath
    raw_folder = root_path + input_filepath

    transform = transforms.Compose([transforms.CenterCrop(300), transforms.ToTensor()])

    if HydraConfig.get()['job']['config_name'] == 'coco_2017':
        annotations_filepath = cfg.hyperparameters.annotations_filepath        
        CocoDataset(raw_folder, (root_path + annotations_filepath), transform)

    elif HydraConfig.get()['job']['config_name'] == 'voc':
        year = cfg.hyperparameters.year
        dataset = cfg.hyperparameters.dataset
                
        try:
            VOCDataset(raw_folder, year, dataset, download=False, transform=transform)
        except RuntimeError:
            VOCDataset(raw_folder, year, dataset, download=True, transform=transform)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
