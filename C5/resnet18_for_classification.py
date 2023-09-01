import matplotlib.ticker as mticker
import matplotlib.ticker as mtick
import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from glob import glob
from random import shuffle, seed
seed(10)

current_dir = os.path.realpath(os.path.dirname(__file__))
cat_dog_dir = os.path.join(current_dir, "..", "data", "CATDOG")
train_data_dir = os.path.join(cat_dog_dir, "training_set", "training_set")
test_data_dir = os.path.join(cat_dog_dir, "test_set", "test_set")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CatsDogs(Dataset):
    def __init__(self, folder):
        cats = glob(folder+'/cats/*.jpg')
        dogs = glob(folder+'/dogs/*.jpg')
        self.fpaths = cats[:500] + dogs[:500]
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.targets = [fpath.split("/")[-1].startswith('dog')
                        for fpath in self.fpaths]

    def __len__(self): return len(self.fpaths)

    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:, :, ::-1])
        im = cv2.resize(im, (224, 224))
        im = torch.tensor(im/255)
        im = im.permute(2, 0, 1)
        im = self.normalize(im)
        return im.float().to(device), torch.tensor([target]).float().to(device)


data = CatsDogs(train_data_dir)

# im, label = data[200]
# plt.imshow(im.permute(1, 2, 0).cpu())
# plt.show()
# print(label)


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer
