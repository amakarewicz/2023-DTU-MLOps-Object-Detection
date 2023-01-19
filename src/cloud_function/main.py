import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['TORCH_HOME'] = '/tmp'
print("Env created")

import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import json
print("Imports done")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", cache_dir='/tmp')
print("Processor loaded")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir='/tmp')
print("Model loaded")

def get_results(pred: dict):
	values_list = [x.tolist() for x in pred.values()]
	keys_list = list(pred.keys())
	return {keys_list[i]: values_list[i] for i in range(len(keys_list))}

def predict(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    # if request.args and 'url' in request.args:
    url = request_json['url']
    image = Image.open(requests.get(url, stream=True).raw)
    print("Image loaded: ", url)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    pred_dict = {'predictions': [get_results(d) for d in pred]}

      # output_prediction_file = os.path.join(output_prediction_dir, "results.json")
      # json_object = json.dumps(pred_dict, indent=4)
      # with open(output_prediction_file, "w") as outfile:
      #     outfile.write(json_object)
    return pred_dict
    #     return request.args.get('message')
    # elif request_json and 'message' in request_json:
    #     return request_json['message']
    # else:
    #     return f'Hello World!'