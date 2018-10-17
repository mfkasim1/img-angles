"""
This is modified file from PyTorch tutorial to take the advantage of deepmk.
"""

import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import deepmk
import deepmk.criteria

# local lib
from dshandler import ImageRotater
from model import get_model

num_workers = 4

# hyper-params
batch_size = 4
model_type = 2 # 1 if the images concat in vertical, 2 if in channel
lr = 1e-6
momentum = 0.9
scheduler_step_size = 7
scheduler_gamma = 0.1
mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]

# data directory
fdir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(fdir, "dataset")

# transformations of the data
data_transforms = {
    "train": transforms.Compose([
        transforms.Normalize(mean_norm, std_norm),
    ]),
    "val": transforms.Compose([
        transforms.Normalize(mean_norm, std_norm),
    ]),
}

# dataset where the data should be loaded
img_size = 224
max_images = {"train": 1000, "val": 100}
image_datasets = {
    x: ImageRotater(root=os.path.join(data_dir, x), imgsize=img_size,
                    max_images=max_images[x], img_transform=data_transforms[x],
                    merge_type=model_type)
    for x in ["train", "val"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
    for x in ["train", "val"]
}

# load the model
model_ft = get_model(model_type, (3, img_size, img_size))

# cross entropy loss for multi classes classifications
criterion = {
    "train": nn.MSELoss(),
    "val": nn.MSELoss(),
}

optimizer_ft = optim.SGD(model_ft.get_trainable_params(), lr=lr,
    momentum=momentum)

# reduce the learning rate by 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
    step_size=scheduler_step_size, gamma=scheduler_gamma)

# train the model
model_ft = deepmk.spv.train(model_ft, dataloaders, criterion, optimizer_ft,
    scheduler=exp_lr_scheduler, num_epochs=125, plot=1, verbose=2,
    save_model_to="angle-resnet.pkl")
