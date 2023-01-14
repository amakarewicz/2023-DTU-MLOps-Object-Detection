import torch
import torch.nn as nn
# from pytorch_lightning import LightningModule

from transformers import DetrImageProcessor, DetrForObjectDetection

class DetrModel(nn.Module): # LightningModule
    def __init__(self, detection_threshold):
        super().__init__()

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.detection_threshold = detection_threshold

    def forward(self, batch): # TODO: change to whole batch, now working on a single image loaded as in modeling_example.ipynb
        inputs = self.processor(images=batch, return_tensors='pt')
        return self.model(**inputs)

    def predict(self, outputs, target_sizes):
        return self.processor.post_process_object_detection(outputs, 
                                                            target_sizes=target_sizes, 
                                                            threshold=self.detection_threshold)

    # TODO: make it work for our model's input
    # def training_step(self, batch, ):
    #     b_input_ids = batch[0]
    #     b_input_mask = batch[1]
    #     b_labels = batch[2]
    #     (loss, _) = self.model(
    #         b_input_ids,
    #         token_type_ids=None,
    #         attention_mask=b_input_mask,
    #         labels=b_labels,
    #     )
    #     self.log("train_loss", loss)
    #     return loss

    # TODO: https://github.com/nielstiben/MLOPS-Project/blob/main/src/models
    # def configure_optimizers(self):