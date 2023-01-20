import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.visualization.categories import VOC_TO_COCO_DICT


class DetrModel(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.processor = DetrImageProcessor.from_pretrained(
            config.model.processor
        )
        self.model = DetrForObjectDetection.from_pretrained(
            config.model.detector
        )
        self.detection_threshold = config.model.detection_threshold

    def forward(self, batch):
        images = [image for (image, _) in batch]
        inputs = self.processor(images=images, return_tensors="pt")
        return self.model(**inputs)

    def training_step(self, batch):
        images = [image for (image, _) in batch]
        if self.config.train.dataset == "coco":
            annotations = [annotation for (_, annotation) in batch]
        if self.config.train.dataset == "voc":
            annotations = [
                annotation["annotation"]["object"] for (_, annotation) in batch
            ]
        labels = [self.get_labels(annot) for annot in annotations]
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model.forward(**inputs, labels=labels)
        loss = outputs[0]
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        wandb.log({"train/loss": loss})
        return loss

    def configure_optimizers(self):
        if self.config.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config.train.lr
            )
        return optimizer

    def predict(self, outputs, target_sizes):
        return self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.detection_threshold,
        )

    # TODO: change if we're using two datasets (different setup for VOC data)
    def get_labels(self, annotations: list):
        labels = {}
        if self.config.train.dataset == "coco":
            labels["class_labels"] = torch.LongTensor(
                [
                    item["category_id"]
                    for item in annotations
                    if "category_id" in item
                ]
            )
            labels["boxes"] = torch.FloatTensor(
                [item["bbox"] for item in annotations if "bbox" in item]
            )  # 'bndbox' for voc
        if self.config.train.dataset == "voc":
            cat_names = [
                item["name"] for item in annotations if "name" in item
            ]
            labels["class_labels"] = torch.LongTensor(
                list(map(VOC_TO_COCO_DICT.get, cat_names))
            )
            boxes = [
                item["bndbox"] for item in annotations if "bndbox" in item
            ]
            num_boxes = [
                dict([a, int(x)] for a, x in b.items()) for b in boxes
            ]
            for b in num_boxes:
                b["width"] = b["xmax"] - b["xmin"]
                b["height"] = b["ymax"] - b["ymin"]
                b.pop("xmax")
                b.pop("ymax")
            labels["boxes"] = torch.FloatTensor(
                [list(d.values()) for d in num_boxes]
            )
        return labels
