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

# from src.data.dataset import DesasterTweetDataModule
from src.models.model import DetrModel
from src.data.fetch_data import VOCDataset

@hydra.main(config_path="./config", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Predicting...")

#     # %% Validate output folder
    output_dir = os.path.join(hydra.utils.get_original_cwd(), config.predict.model_output_dir)
    if not os.path.isdir(output_dir):
        raise Exception(
            'The "model_output_dir" path ({}) could not be found'.format(output_dir)
        )

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

    model = DetrModel() # (output_config)
#     model.load_from_checkpoint(output_checkpoint_latest_path, config=output_config)
    outputs = model.forward() # TODO: add data
#     # %% Load data module and use Validation data
#     data_module = DesasterTweetDataModule(
#         os.path.join(hydra.utils.get_original_cwd(), config.data.path),
#         batch_size=config.train.batch_size,
#     )
#     data_module.setup()
#     data = data_module.valset

    # %% Predict and save to output directory
    output_prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(output_prediction_dir, exist_ok=True)
    start_time = time()
    pred = model.predict(outputs)
#     y_pred_np = y_pred.logits.detach().numpy()
    output_prediction_file = os.path.join(output_prediction_dir, "predictions.json")
    # np.savetxt(output_prediction_file, y_pred_np, delimiter=",")

    # Serializing json
    json_object = json.dumps(pred, indent=4)
    
    # Writing to sample.json
    with open(output_prediction_file, "w") as outfile:
        outfile.write(json_object)

    logger.info(
        "Predictions generated in {} seconds, saved to {}".format(
            round(time() - start_time), output_prediction_file
        )
    )


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
