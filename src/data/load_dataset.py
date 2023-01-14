from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

class CocoDataset(datasets.CocoDetection):
    def __init__(self,
                 root: str,
                 annFile: str,
                 transform=None,
                 target_transform=None) -> None:
        super().__init__(root, annFile, transform, target_transform)

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, target

class VOCDataset(datasets.VOCDetection):
    """Fetch Visual Object Classes (VOC) data set  <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self,
                 root: str, 
                 year: str,
                 image_set: str,
                 download: bool=False,
                 transform=None,
                 target_transform=None) -> None:
        super().__init__(root, year, image_set, download, transform, target_transform)

        self.data = datasets.VOCDetection(root = root, year = year, image_set=image_set, download=download, transform=transform)
        
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        # do whatever you want
        return img, target
    
    def show(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    def make_dataloader(self, batch_size: int, shuffle: bool = True):
        return DataLoader(self.data, batch_size, shuffle=True, collate_fn=lambda x: x )

class LoadImages():
    def __init__(self,
    paths: dict = {
        'voc': 'data/raw/voc',
        'coco': 'data/raw/coco/images/val2017/',
        'coco_annotations': 'data/raw/coco/annotations/instances_val2017.json'
        },
    voc_year: int = 2007,
    voc_dataset: str = 'val'):
        self.paths = paths
        self.voc_year = str(voc_year)
        self.voc_dataset = voc_dataset
        self.transform = transforms.Compose([transforms.CenterCrop(300), transforms.ToTensor()])

    def get_target_sizes(self, images: list):
        transform = transforms.ToPILImage()
        images_pil = [transform(image) for image in images]
        return torch.tensor([image.size[::-1] for image in images_pil])

    def load_voc(self, batch_size: int): 
        # Batch size only for testing! In the future set to dataset size
        dataset = VOCDataset(self.paths['voc'], self.voc_year, self.voc_dataset, download=False, transform=self.transform)
        # batch_size = len(dataset)
        dataloader = dataset.make_dataloader(batch_size=batch_size)

        images = [image for (image, _) in next(iter(dataloader))]
        annotations = [annotation for (_, annotation) in next(iter(dataloader))]
        target_sizes = self.get_target_sizes(images)

        return images, annotations, target_sizes
