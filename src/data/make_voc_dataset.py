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

from load_dataset import VOCDataset

if __name__ == '__main__':
       
    repo_root = Path(__file__).resolve().parents[2] 
    raw_folder = os.path.join(repo_root, 'data', 'raw', 'voc')
    
    # Transform applied on input and target
    transform = transforms.Compose([transforms.CenterCrop(300), transforms.ToTensor()])
    try:
        voc = VOCDataset(raw_folder, '2007', 'val', download=False, transform=transform)
    except RuntimeError:
        voc = VOCDataset(raw_folder, '2007', 'val', download=True, transform=transform)
    
    print(voc.data)
    
    #data  = torchvision.datasets.VOCDetection(root = raw_folder, year = '2012', image_set = 'val', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(voc.data, batch_size=4, shuffle=True, collate_fn=lambda x: x )
    print(len(next(iter(trainloader))))

    train_img, train_label, val_img, val_label = next(iter(trainloader))
    print(val_img)
    
    #voc.show(torchvision.utils.make_grid(images, padding=20))
    #print(data) 
    #print(data) 