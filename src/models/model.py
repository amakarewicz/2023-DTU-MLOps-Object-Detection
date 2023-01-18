from omegaconf import DictConfig
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import wandb

from transformers import DetrImageProcessor, DetrForObjectDetection

class DetrModel(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.processor = DetrImageProcessor.from_pretrained(config.model.processor)
        self.model = DetrForObjectDetection.from_pretrained(config.model.detector)
        self.detection_threshold = config.model.detection_threshold

    def forward(self, batch): 
        images = [image for (image, _) in batch]
        inputs = self.processor(images=images, return_tensors='pt')
        return self.model(**inputs)

    def training_step(self, batch):
        images = [image for (image, _) in batch]
        annotations = [annotation for (_, annotation) in batch]
        labels = [self.get_labels(annot) for annot in annotations]
        inputs = self.processor(images=images, return_tensors='pt')
        outputs = self.model.forward(**inputs, labels=labels)
        loss = outputs[0]
        self.log("train_loss", loss)
        wandb.log({"train/loss": loss})
        return loss

    def configure_optimizers(self):
        if self.config.train.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
        return optimizer

    def predict(self, outputs, target_sizes):
        return self.processor.post_process_object_detection(outputs, 
                                                            target_sizes=target_sizes, 
                                                            threshold=self.detection_threshold)

    # TODO: change if we're using two datasets (different setup for VOC data)
    def get_labels(self, annotations: list):
        labels = {}
        labels['class_labels'] = torch.LongTensor([item['category_id'] for item in annotations if 'category_id' in item]) 
        labels['boxes'] = torch.FloatTensor([item['bbox'] for item in annotations if 'bbox' in item]) #'bndbox' for voc
        return labels

    # def save_jit(self, file: str = "model.pt") -> None:
    #     # token_len = self.config["build_features"]["max_sequence_length"]
    #     # tokens_tensor = torch.ones(1, token_len).long()
    #     # mask_tensor = torch.ones(1, token_len).float()
    #     script_model = torch.jit.trace(self) #, [tokens_tensor, mask_tensor])
    #     script_model.save(file)
