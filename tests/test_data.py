#TestData
import torch
import pytest
import os
from omegaconf import DictConfig
import hydra

from src.data.load_dataset import *
from hydra import initialize, compose 

from tests import _PATH_DATA
from tests import _PATH_DATA, _PROJECT_ROOT


@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA),
    reason="Data files not found",
)

def test_coco_data():
    path = _PATH_DATA + "/raw/coco/annotations"
    dir = os.listdir(path)
    assert len(dir)>0


def test_data_size():
     with initialize("../src/conf"):
        cfg = compose(config_name="default_config.yaml")
        loader=LoadImages()
        dataloader = loader.get_dataloader(dataset=cfg.predict.dataset, batch_size=cfg.predict.batch_size, shuffle=False)
        batch = next(iter(dataloader))
        assert len(batch) == 5


