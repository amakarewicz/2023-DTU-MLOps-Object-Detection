import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CocoDataset(datasets.CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform)

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, target


class VOCDataset(datasets.VOCDetection):
    """Fetch Visual Object Classes (VOC) data set
    <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to annotation file.
        transform (callable, optional): A function/transform that
        takes in an PIL image and returns a transformed version.
        E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that
        takes in the target and transforms it.
        transforms (callable, optional): A function/transform that
        takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        year: str,
        image_set: str,
        download: bool = False,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(
            root, year, image_set, download, transform, target_transform
        )

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, target

    def show(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")


class LoadImages:
    def __init__(
        self,
        root_dir: str = os.getcwd(),
        voc_year: int = 2007,
        voc_dataset: str = "val",
    ):
        self.root_dir = root_dir
        self.paths = {
            "voc": os.path.join(root_dir, "data", "raw", "voc"),
            "coco": os.path.join(
                root_dir, "data", "raw", "coco", "images", "val2017"
            ),
            "coco_annotations": os.path.join(
                root_dir,
                "data",
                "raw",
                "coco",
                "annotations",
                "instances_val2017.json",
            ),
        }
        self.voc_year = str(voc_year)
        self.voc_dataset = voc_dataset
        self.transform = transforms.ToTensor()

    def get_target_sizes(self, batch: tuple):
        images = [image for (image, _) in batch]
        transform = transforms.ToPILImage()
        images_pil = [transform(image) for image in images]
        return torch.tensor([image.size[::-1] for image in images_pil])

    def load_data(self, dataset: str, batch_size: int, shuffle: bool = True):
        """
        Description: Loads images, annotations,
        and target sizes of the selected dataset in a tuple.
        Args:
            dataset: 'coco' to load coco data, 'voc' to load voc data
            batch_size: Restricts the size of the data.
            Only for testing purposes!
        """

        dataloader = self.get_dataloader(dataset, batch_size, shuffle)
        images = [image for (image, _) in next(iter(dataloader))]
        annotations = [
            annotation for (_, annotation) in next(iter(dataloader))
        ]
        target_sizes = self.get_target_sizes(images)

        return images, annotations, target_sizes

    def get_dataloader(
        self, dataset: str, batch_size: int, shuffle: bool = True
    ):
        assert dataset == "coco" or dataset == "voc"

        # batch_size = len(dataset)
        if dataset == "voc":
            dataset = VOCDataset(
                self.paths["voc"],
                self.voc_year,
                self.voc_dataset,
                download=False,
                transform=self.transform,
            )
        elif dataset == "coco":
            print(self.paths["coco"])
            print(self.paths["coco_annotations"])
            dataset = CocoDataset(
                self.paths["coco"],
                self.paths["coco_annotations"],
                self.transform,
            )

        return DataLoader(
            dataset, batch_size, shuffle=shuffle, collate_fn=lambda x: x
        )
