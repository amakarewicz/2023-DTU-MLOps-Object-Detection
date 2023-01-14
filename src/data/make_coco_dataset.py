# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms

import hydra
from hydra.core.hydra_config import HydraConfig
import torch

class CocoDataset(datasets.CocoDetection):
    def __init__(self,
                 root: str,
                 annFile: str,
                 transform=None,
                 target_transform=None) -> None:
        super().__init__(root, annFile, transform, target_transform)

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        # do whatever you want
        return img, target

# TODO VOC
# class VOCDataset(datasets.VOCDetection):
#     def __init__(self,
#                  root: str,
#                  annFile: str,
#                  transform=None,
#                  target_transform=None) -> None:
#         super().__init__(root, annFile, transform, target_transform)

#     def __getitem__(self, index: int):
#         img, target = super().__getitem__(index)
#         # do whatever you want
#         return img, target

@hydra.main(config_path='../conf/dataset', config_name='coco_2017')
def main(cfg):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    root_path = HydraConfig.get().runtime.cwd
    input_filepath = cfg.hyperparameters.input_filepath
    annotations_filepath = cfg.hyperparameters.annotations_filepath
    batchsize = cfg.hyperparameters.batchsize
    shuffle = cfg.hyperparameters.shuffle
    output_filepath = cfg.hyperparameters.output_filepath

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CocoDataset((root_path + input_filepath), (root_path + annotations_filepath), transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

    image, label = next(iter(dataloader))
    # print(label)

    # TODO VOC
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    
    # voc_data = datasets.VOCDetection((root_path + "/data/raw/voc/val"), '2007', 'val', download = True, transforms = transform)
    # dataloader = torch.utils.data.DataLoader(voc_data, batch_size=batchsize, shuffle=shuffle)

    # print(next(iter(dataloader)))
    # print(label)

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
