import glob
import logging
import os
from pathlib import Path
from time import time
import json
import hydra
import numpy as np
# from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os

# For drawing bounding boxes.
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as transforms

import torch
from PIL import Image
import requests
import json
from hydra.core.hydra_config import HydraConfig
from src.data.load_dataset import *
from src.visualization.coco_categories import COCO_INSTANCE_CATEGORY_NAMES

@hydra.main(config_path="../conf", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Visualizing...")

    # how to get the original image !!!
    # results - from json
    root_dir = HydraConfig.get().runtime.cwd
    
    images_dir_path = os.path.join('E:/mlops/', config.visualize.images_input_dir)
    image_path = os.path.join(images_dir_path, os.listdir(images_dir_path)[config.visualize.image_no])

    results_path = os.path.join(root_dir, config.visualize.results_input_dir, 'predictions/results.json')
    with open(results_path, 'r') as j:
        batch_results = json.loads(j.read())
    results = batch_results['predictions'][config.visualize.image_no]
    pred_scores = results['scores'] 
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in results['labels']]
    pred_bboxes = results['boxes']
    pred_classes = pred_classes[:len(pred_bboxes)]

    colors = np.random.randint(0, 255, size=(len(pred_bboxes), 3))
    colors = [tuple(color) for color in colors]

    image_np = np.array(Image.open(image_path))
    image_transposed = np.transpose(image_np, [2, 0, 1])
    int_input = torch.tensor(image_transposed)

    result_with_boxes = draw_bounding_boxes(
        image=int_input, 
        boxes=torch.tensor(pred_bboxes), width=4, 
        colors=colors,
        labels=pred_classes,
        # fill=True
    )

    torchvision.transforms.ToPILImage()(result_with_boxes).show()
    # TODO: wrong assignment (results - image)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()