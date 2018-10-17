import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import deepmk
import deepmk.criteria
import maleo
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

        # crop resize image
        crop_resize_transform = transforms.RandomResizedCrop(self.imgsize)
        img = crop_resize_transform(img)

        # decide the angle to rotate and rotate the image
        angle1, angle2 = np.random.random(2) * 360
        img1 = img.rotate(angle1)
        img2 = img.rotate(angle2)

        # convert to torch tensor
        totensor = transforms.ToTensor()
        img1 = totensor(img1)
        img2 = totensor(img2)

        # calculate the target (i.e. the relative angle between 2 images)
        ang = (angle2 - angle1) % 360 # the angle must be [0, 360)

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
        target = torch.FloatTensor([ang/360.])
        return imgcat, target

class Model1(nn.Module):
    """
    This model handles if the two images are concatenated in vertical direction.
    """
    def __init__(self, img_shape):
        super(Model1, self).__init__()
        # load the pretrained resnet18
        model_ft = models.resnet18(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False

        # try propagating a dummy image, just to see the output size
        model_ft.fc = nn.ReLU()
        dum_out = model_ft.forward(Variable(torch.randn(1, *img_shape)))
        in_features = dum_out.numel()

        # replace the fully connected layer to a new layer with 2 outputs
        model_ft.fc = nn.Linear(in_features, 1)
        self.model = model_ft

    def forward(self, x):
        return self.model.forward(x)

    def get_trainable_params(self):
        return self.model.fc.parameters()

class Model2(nn.Module):
    """
    This model handles if the two images are concatenated in channel.
    """
    def __init__(self, img_shape):
        super(Model2, self).__init__()
        # load the pretrained resnet18
        self.model1 = models.resnet18(pretrained=True)
        for param in self.model1.parameters():
            param.requires_grad = False

        # try propagating a dummy image, just to see the output size
        self.model1.fc = nn.ReLU()
        dum_out = self.model1.forward(Variable(torch.randn(1, *img_shape)))
        in_features = dum_out.numel()

        # get the model 2
        self.model2 = nn.Linear(2*in_features, 1)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        img1, img2 = x[:,:3,:,:], x[:,3:,:,:] # separate the channels
        x1 = self.model1(img1)
        x2 = self.model1(img2)
        x12 = torch.cat((x1.view(x1.size(0),-1), x2.view(x2.size(0),-1)), dim=1)
        return self.model2(x12)

    def get_trainable_params(self):
        return self.model2.parameters()

def get_model(model_no, img_shape):
    if model_no == 1:
        shape = list(img_shape)
        shape[1] *= 2 # double in vertical direction
        model = Model1(shape)
    elif model_no == 2:
        model = Model2(img_shape)
    return model

# hyper-params
def train(
        batch_size = 4,
        lr = 1e-6,
        momentum = 0.9,
        scheduler_step_size = 7,
        scheduler_gamma = 0.1,
        mean_norm = [0.485, 0.456, 0.406],
        std_norm = [0.229, 0.224, 0.225],
        ):
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
                        max_images=max_images[x],
                        img_transform=data_transforms[x],
                        merge_type=model_type)
        for x in ["train", "val"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=batch_size,
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
        scheduler=exp_lr_scheduler, num_epochs=25, plot=0, verbose=2,
        save_model_to="angle-resnet.pkl")

    value = deepmk.spv.validate(model_ft, dataloaders["val"], criterion["val"])
    return value

def main():
    op = maleo.Solver("test")
    op.set_algorithm(maleo.alg.CMAES(max_fevals=1000, populations=16))
    op.add_resource(maleo.LocalResource(
        max_jobs=8, scheduler="taskset 8 9 10 11 12 13 14 15"))
    op.set_function(train)

    # hyperparams
    op.add_variable(maleo.Scalar('batch_size', lbound=4, ubound=32, is_integer=True))
    op.add_variable(maleo.Scalar('lr', lbound=1e-10, ubound=1, logscale=True))
    op.add_variable(maleo.Scalar('momentum', lbound=0.1, ubound=1))
    op.add_variable(maleo.Scalar('scheduler_step_size', lbound=5, ubound=20, is_integer=True))
    op.add_variable(maleo.Scalar('scheduler_gamma', lbound=0.001, ubound=1))
    op.add_variable(maleo.Vector('mean_norm', size=3, lbounds=0, ubounds=1))
    op.add_variable(maleo.Vector('std_norm', size=3, lbounds=0.1, ubounds=2))

    # Run the process. If a file with the same name appears (e.g. "test" in
    # this case), it will read the file and resume from where it stops.
    res = op.run()

    # print the output
    op.print_result()

if __name__ == "__main__":
    main()
