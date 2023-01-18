import glob
import logging
import os
from pathlib import Path
import time
import json
import hydra
import numpy as np
# from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from src.models.model import DetrModel
from src.data.load_dataset import *
from hydra.core.hydra_config import HydraConfig

def get_results(pred: dict):
	values_list = [x.tolist() for x in pred.values()]
	keys_list = list(pred.keys())
	return {keys_list[i]: values_list[i] for i in range(len(keys_list))}

@hydra.main(config_path="../conf", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Predicting...")
    # %% Validate output folder
    # output_dir = os.path.join(hydra.utils.get_original_cwd(), config.predict.model_output_dir)
    # if not os.path.isdir(output_dir):
    #     raise Exception(
    #         'The "model_output_dir" path ({}) could not be found'.format(output_dir)
    #     )
    
    loader = LoadImages(root_dir = HydraConfig.get().runtime.cwd)
    dataloader = loader.get_dataloader(config.predict.dataset, config.predict.batch_size, shuffle=False)
    batch = next(iter(dataloader))

#     # %% Load local config in output directory
#     output_config_path = os.path.join(output_dir, ".hydra", "config.yaml")
#     output_config = omegaconf.OmegaConf.load(output_config_path)

#     # %% Load model
#     output_checkpoints_paths = os.path.join(
#         output_dir, "lightning_logs", "version_0", "checkpoints", "*.ckpt"
#     )
#     output_checkpoint_latest_path = sorted(
#         filter(os.path.isfile, glob.glob(output_checkpoints_paths))
#     )[-1]

    model = DetrModel(config)
#     model.load_from_checkpoint(output_checkpoint_latest_path, config=output_config)
    outputs = model.forward(batch)

    # %% Predict and save to output directory
    output_prediction_dir = os.path.join(os.getcwd(), "predictions")
    os.makedirs(output_prediction_dir, exist_ok=True)

    start_time = time.time()
    pred = model.predict(outputs, target_sizes=loader.get_target_sizes(batch))
    pred_dict = {'predictions': [get_results(d) for d in pred]}
    
    output_prediction_file = os.path.join(output_prediction_dir, "results.json")

    json_object = json.dumps(pred_dict, indent=4)
    with open(output_prediction_file, "w") as outfile:
        outfile.write(json_object)

    logger.info(
        "Predictions generated in {} seconds, saved to {}".format(
            round(time.time() - start_time), output_prediction_file
        )
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
