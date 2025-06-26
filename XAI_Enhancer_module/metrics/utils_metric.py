import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image
import glob


mean = [0.6380, 0.3422, 0.2275]
std = [0.2448, 0.2060, 0.1710]

# Dummy class to store arguments
class Dummy():
    pass


# Function that opens image from disk, normalizes it and converts to tensor
def read_tensor(image, model_name):
    image = Image.open(image)
    img_sz = 224
    if model_name in ('b4'):
        img_sz = 384
    tfms = transforms.Compose([transforms.Resize((img_sz, img_sz)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = mean, std = std)])
    # tfms = transforms.Compose([transforms.Resize((img_sz, img_sz)), transforms.ToTensor()])
    image = tfms(image)
    image = torch.unsqueeze(image, 0)    
    return image


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)

#fn to fetch test images
def create_test_batch(model_name):
    img_size = 224
    if model_name in ('b4'):
        img_size = 384
    test_images = torch.empty((6, 3, img_size, img_size))
    path = [i for i in glob.glob('./Test_images_IBS' + '/*')]
    path.sort()
    for it, i in enumerate(path):
        image = read_tensor(i, model_name)
        test_images[it] = image
    return test_images

#fn to fetch mappings
def create_mask_batch(model, cam):
    test_mask_path = f'./output_gray_map/{model}_{cam}'
    img_size = 224
    if model in ('b4'):
        img_size = 384

    mask_batch = np.empty((6, img_size, img_size))
    for it, i in enumerate(glob.glob(test_mask_path + '/*')):
        mask = np.load(i)
        mask_name = i.split('/')[-1]
        if mask_name.startswith('0_0'):
            mask_batch[2] = mask
        elif mask_name.startswith('0_1'):
            mask_batch[4] = mask
        elif mask_name.startswith('0_2'):
            mask_batch[1] = mask
        elif mask_name.startswith('0_3'):
            mask_batch[3] = mask
        elif mask_name.startswith('0_4'):
            mask_batch[0] = mask
        elif mask_name.startswith('0_5'):
            mask_batch[5] = mask
    
    return mask_batch

# Given label number returns class name
def get_class_name(c):
    if c==0:
        return "IBS"
    return "Normal"

# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
