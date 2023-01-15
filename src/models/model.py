import torch
import torch.nn as nn
# from pytorch_lightning import LightningModule

from transformers import DetrImageProcessor, DetrForObjectDetection

class DetrModel(nn.Module): # LightningModule
    def __init__(self):
        super().__init__()

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def forward(self, image): # TODO: change to whole batch, now working on a single image loaded as in modeling_example.ipynb
        inputs = self.processor(images=image, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs

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