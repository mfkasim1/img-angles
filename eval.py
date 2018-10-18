import os
import shutil
import argparse
import torch
from torchvision import transforms
from PIL import Image

from model import get_model
from dshandler import ImageRotater

model_type = 2 # 1 if the images concat in vertical, 2 if in channel

def eval(img1_path, img2_path):
    """
    Calculate and return the relative angle between two images.
    """
    # load the model
    img_size = 224
    model = get_model(model_type, (3, img_size, img_size))
    model.cuda()
    model.load_state_dict(torch.load("angle-resnet-wts.pkl"))

    # transforms the two images
    img1 = Image.open(os.path.join(img1_path))
    img2 = Image.open(os.path.join(img2_path))
    mean_norm = [0.15192785484028182,0.10317494370081018,0.5673184447640081]
    std_norm = [0.17425616586457335,1.1235511510118605,0.30659749320043583]
    transform = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean_norm, std_norm)
    ])
    img1 = transform(img1)
    img2 = transform(img2)
    # check the channel of the images (some image only have 1 channel)
    if img1.shape[0] == 1:
        img1 = torch.cat((img1, img1, img1), dim=0)
        img2 = torch.cat((img2, img2, img2), dim=0)

    if model_type == 1:
        # concat img1 and img2 in vertical direction (dimension #1)
        dimcat = 1
    elif model_type == 2:
        # concat img1 and img2 in channel (dimension #0)
        dimcat = 0
    imgcat = torch.cat((img1, img2), dim=dimcat)

    # evaluate it using our model
    inp = imgcat.unsqueeze(0).cuda()
    norm_angle = model.forward(inp)
    angle = norm_angle * 360
    return float(angle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1_path", type=str)
    parser.add_argument("img2_path", type=str)
    args = parser.parse_args()
    angle = eval(args.img1_path, args.img2_path)
    print("The relative angle is %f degree" % angle)

if __name__ == "__main__":
    main()
