import torch 
import torchvision
import PIL.Image as Image
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from pathlib import Path


class VOCDataset(torchvision.datasets.VOCDetection):
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

        self.data = torchvision.datasets.VOCDetection(root = root, year = year, image_set=image_set, download=download, transform=transform)
        
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        # do whatever you want
        return img, target
    
    def show(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


if __name__ == '__main__':
       
    repo_root = Path(__file__).resolve().parents[2] 
    raw_folder = os.path.join(repo_root, 'data', 'raw', 'voc')
    
    # Transform applied on input and target
    transform = transforms.Compose([transforms.CenterCrop(300), transforms.ToTensor()])
    try:
        voc = VOCDataset(raw_folder, '2007', 'val', download=False, transform=transform)
    except RuntimeError:
        voc = VOCDataset(raw_folder, '2007', 'val', download=True, transform=transform)
    
    #print(voc.data)
    
    #data  = torchvision.datasets.VOCDetection(root = raw_folder, year = '2012', image_set = 'val', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(voc.data, batch_size=1, shuffle=True, collate_fn=lambda x: x )
    print(next(iter(trainloader)))

    train_img, train_label = next(iter(trainloader))
    #print(train_label)
    
    #voc.show(torchvision.utils.make_grid(images, padding=20))
    #print(data) 
    #print(data) 