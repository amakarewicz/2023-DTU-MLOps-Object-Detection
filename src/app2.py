import glob
import logging
import os
from pathlib import Path
import time
import json
import hydra
import numpy as np
# from dotenv import find_dotenv, load_dotenv
from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from omegaconf import DictConfig
from src.models.model import DetrModel
from src.data.load_dataset import *
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from hydra import compose
#from config import api_config, model_config



def get_results(pred: dict):
    values_list = [x.tolist() for x in pred.values()]
    keys_list = list(pred.keys())
    return {keys_list[i]: values_list[i] for i in range(len(keys_list))}

@hydra.main(config_path="conf", config_name="default_config")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Predicting...")
    # Validate output folder
    # output_dir = os.path.join(hydra.utils.get_original_cwd(), config.predict.model_output_dir)
    # if not os.path.isdir(output_dir):
    #     raise Exception(
    #         'The "model_output_dir" path ({}) could not be found'.format(output_dir)
    #     )
    
    loader = LoadImages()
    dataloader = loader.get_dataloader(config.predict.dataset, config.predict.batch_size, shuffle=False)
    batch = next(iter(dataloader))

#     #Load local config in output directory
#     output_config_path = os.path.join(output_dir, ".hydra", "config.yaml")
#     output_config = omegaconf.OmegaConf.load(output_config_path)

#     #Load model
#     output_checkpoints_paths = os.path.join(
#         output_dir, "lightning_logs", "version_0", "checkpoints", "*.ckpt"
#     )
#     output_checkpoint_latest_path = sorted(
#         filter(os.path.isfile, glob.glob(output_checkpoints_paths))
#     )[-1]

    model = DetrModel(config)
#     model.load_from_checkpoint(output_checkpoint_latest_path, config=output_config)
    outputs = model.forward(batch)

    #Predict and save to output directory
    output_prediction_dir = os.path.join(os.getcwd(), "predictions")
    os.makedirs(output_prediction_dir, exist_ok=True)

    start_time = time.time()
    pred = model.predict(outputs, target_sizes=loader.get_target_sizes(batch))
    pred_dict = {'predictions': [get_results(d) for d in pred]}
    
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    output_prediction_file = os.path.join(output_prediction_dir, "results.json")

    json_object = json.dumps(pred_dict, indent=4)
    print(json_object)
    with open(output_prediction_file, "w") as outfile:
        outfile.write(json_object)

    logger.info(
        "Predictions generated in {} seconds, saved to {}".format(
            round(time.time() - start_time), output_prediction_file
        )
    )

    return json_object

# def get_results(pred: dict):
# 	values_list = [x.tolist() for x in pred.values()]
# 	keys_list = list(pred.keys())
# 	return {keys_list[i]: values_list[i] for i in range(len(keys_list))}


# #@hydra.main(config_path="../conf", config_name="default_config.yaml")
# def main(config: DictConfig):
#     logger = logging.getLogger(__name__)
#     logger.info("Predicting...")
    
#     loader = LoadImages()
#     dataloader = loader.get_dataloader(api_config['dataset'], api_config['batch_size'], shuffle=False)
#     batch = next(iter(dataloader))

#     model = DetrModel(config)
# #     model.load_from_checkpoint(output_checkpoint_latest_path, config=output_config)
#     outputs = model.forward(batch)

#     # %% Predict and save to output directory
#     output_prediction_dir = os.path.join(os.getcwd(), "predictions")
#     os.makedirs(output_prediction_dir, exist_ok=True)

#     start_time = time.time()
#     pred = model.predict(outputs, target_sizes=loader.get_target_sizes(batch))
#     pred_dict = {'predictions': [get_results(d) for d in pred]}
    
#     time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
#     output_prediction_file = os.path.join(output_prediction_dir, "results.json")

#     json_object = json.dumps(pred_dict, indent=4)
#     with open(output_prediction_file, "w") as outfile:
#         outfile.write(json_object)

#     logger.info(
#         "Predictions generated in {} seconds, saved to {}".format(
#             round(time.time() - start_time), output_prediction_file
#         )
#     )

#     return pred_dict 





######
# APP 
# class Request(BaseModel):
#     text: str   


# pred_model = predict() 
# print(pred_model.main)


app = FastAPI()


# @app.get("/")
# def read_root():
#    return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int):
#    return {"item_id": item_id}

#main()

# # allow CORS requests from any host so that the JavaScript can communicate with the server
# app.add_middleware(
#     CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API root
@app.get("/")
def get_root():
    return "This is the RESTful API for MLOps test"

database = {}
# POST endpoint with path '/predict'
@app.post("/predict/")
def predictions():
    predictions = main()
    if not database:
        with open('database.csv', "a") as file:
            file.write(predictions)
    return "preds made"
    # return {
    #     "predictions": 'hellooow' #predictions
    # }

from fastapi import FastAPI, Response
@app.get("/predictions/")
def read_item():
    return database
# async def predictions():
#     #predictions = pred_model.main
#     return {
#         "predictions": 'hellooow' #predictions
#     }
    
# class KeywordResponse(BaseModel):
#     response: dict[str, list[str]]
      
# # class AllModelsResponse(BaseModel):
# #     original: str
# #     paraphrased: ParagraphResponse
# #     name_entities: NERResponse
# #     summarized: ParagraphResponse
# #     keyword_synonyms: KeywordResponse

# # @app.post("/predict", response_model=AllModelsResponse)
# # async def predict(request: Request):
# #     paraphrased_text = ParagraphResponse(text=paraphrasing_pipeline(request.text))
# #     ner_text = NERResponse(render_data=ner_pipeline(request.text))
# #     summarized_text = ParagraphResponse(text=summarization_pipeline(request.text))
# #     keyword_synonyms = KeywordResponse(response=keyword_pipeline.get_synonyms_for_keywords(request.text))
# #     return AllModelsResponse(
# #         original=request.text, paraphrased=paraphrased_text, 
# #         name_entities=ner_text, summarized=summarized_text,
# #         keyword_synonyms=keyword_synonyms
# #     )
    
