import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

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
