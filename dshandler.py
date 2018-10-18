import os
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image

class ImageRotater(data.Dataset):
    """
    Dataset handler where it loads an image and rotate it with a random angle
    around the centre of the image.
    """
    def __init__(self, root, imgsize=224, max_images=None, img_transform=None,
            merge_type=1):
        self.root = root
        self.imgsize = imgsize
        self.img_transform = img_transform

        # list all the jpg filenames
        self.fnames = [fname for fname in os.listdir(root) if ".jpg" in fname]
        if max_images is None:
            self.max_images = len(self.fnames)
        else:
            self.max_images = np.maximum(max_images, len(self.fnames))
        self.merge_type = merge_type

    def __len__(self):
        return self.max_images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, angle).
                image is a torch tensor of the concatenated image1 and image2.
                target is the relative angle between image1 and image2.
                Both are FloatTensor.
        """
        # load the image
        # i = np.random.randint(len(self.fnames))
        fname = self.fnames[index]
        img = Image.open(os.path.join(self.root, fname))

        # crop the image
        transform = transforms.CenterCrop(self.imgsize)
        img = transform(img)

        # rotate the image
        angle = np.random.randint(360)
        rot_img = img.rotate(angle)

        # convert to torch tensor
        totensor = transforms.ToTensor()
        img1 = totensor(img)
        img2 = totensor(rot_img)

        # apply the user-defined transform for the images
        if self.img_transform != None:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)

        # check the channel of the images (some image only have 1 channel)
        if img1.shape[0] == 1:
            img1 = torch.cat((img1, img1, img1), dim=0)
            img2 = torch.cat((img2, img2, img2), dim=0)

        if self.merge_type == 1:
            # concat img1 and img2 in vertical direction (dimension #1)
            dimcat = 1
        elif self.merge_type == 2:
            # concat img1 and img2 in channel (dimension #0)
            dimcat = 0
        imgcat = torch.cat((img1, img2), dim=dimcat)

        # normalize the target
        target = torch.FloatTensor([angle/360.])
        return imgcat, target
