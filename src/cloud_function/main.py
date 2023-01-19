import os
import pickle
import gcsfs
import requests
from PIL import Image
from categories import COCO_CATEGORIES

os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['TORCH_HOME'] = '/tmp'

from transformers import DetrImageProcessor
import torch

fs = gcsfs.GCSFileSystem(project='DTU-MLOps-Object-Detection')

with fs.open("od-model-checkpoints/deployable_model.pkl", "rb") as file:
    model = pickle.load(file)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",
                                               cache_dir='/tmp')


class Output():
    def __init__(self, logits, pred_boxes):
        self.logits = logits
        self.pred_boxes = pred_boxes


def get_results(pred: dict):
    values_list = [x.tolist() for x in pred.values()]
    keys_list = list(pred.keys())
    return {keys_list[i]: values_list[i] for i in range(len(keys_list))}


def predict(request):

    request_json = request.get_json()
    if "url" in request_json.keys():
        url = request_json['url']
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        out = Output(outputs[0], outputs[1])
        t_sizes = torch.tensor([image.size[::-1]])
        pred = processor.post_process_object_detection(out,
                                                       target_sizes=t_sizes,
                                                       threshold=0.9)[0]
        results = get_results(pred)
        results['labels'] = [COCO_CATEGORIES[i] for i in results['labels']]

        return results

    else:
        return 'Invalid request. Provide image url using "url" key.'
        