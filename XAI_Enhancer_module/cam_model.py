from typing import Any
import numpy as np
import pandas as pd 
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from torch.utils.data import Dataset
import cv2
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from XAI_Enhancer_module.model_utils import get_device, transformations

device = get_device()

class camDataset(Dataset):
    def __init__(self, cam_name, model_name, image_filepaths, labels, model, conv_list):
        self.cam = cam_name
        self.model_name = model_name
        self.image_filepaths = image_filepaths
        self.labels = labels
        self.tfms = transformations
        self.model = model
        self.conv_list = conv_list

    def __len__(self):
        return len(self.image_filepaths)
    
    def grad_cam(self, image, label, target_layer):
        
        cam = self.cam(self.model, target_layer)
        targets = [ClassifierOutputTarget(label)]
        modified_output, grayscale_cam = cam(input_tensor=image, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        return image, grayscale_cam, modified_output
    
    def class_to_label(self, label):
        if label == 'IBS_1':
            pred = 0
        else:
            pred = 1
        return pred
    
    def hook_and_cook(self, target_layer, model, input_tensor, intermediate_output):
        def hook(module, input, output):
            output = intermediate_output
            return output
            
        hook_1 = target_layer[0].register_forward_hook(hook)

        modified_output = model(input_tensor)

        hook_1.remove()

        return modified_output
    
    def forward_pass_actual(self, input, target_class):
        actual_output = self.model(input)
        actual_output = actual_output[0].cpu().data.numpy()[target_class]
        return actual_output
    
    def transform_image(self, image):
        img_size = [384 if self.model_name.startswith("b4") else 224]
        image = cv2.resize(image, (img_size[0], img_size[0]))
        image = image/255.0
        image = self.tfms(image)
        image = image.float()
        input_tensor = torch.unsqueeze(image, dim = 0)
        return input_tensor
    
    def euclidean_diff(self, actual_output, modified_output):
        difference = torch.sqrt(torch.sum(torch.pow(torch.subtract(actual_output, modified_output), 2), dim=1)) 
        return difference.detach().cpu().numpy()

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        image = plt.imread(image_filepath)
        label = self.labels[idx]
        pred_label = self.class_to_label(label)
        input_tensor = self.transform_image(image)
        actual_output = self.forward_pass_actual(input_tensor, pred_label) #being calculated repetitively [try to sort this]
        sum_of_all_values = 0
        weighted_final_cam_image = 0

        for layer in self.conv_list:
            target_layer = [dict(*[self.model.named_modules()])[layer]]
            image, cam_image, intermediate_output = self.grad_cam(input_tensor, pred_label, target_layer)
            intermediate_map = torch.from_numpy(intermediate_output)
            cam_image = torch.from_numpy(cam_image)
            modified_output = self.hook_and_cook(target_layer, self.model, image, intermediate_map)
            abs_diff = self.euclidean_diff(actual_output, modified_output)
            sum_of_all_values += abs_diff
            weighted_final_cam_image += cam_image * abs_diff

        weighted_final_cam_image -= weighted_final_cam_image.min()
        weighted_final_cam_image /= (1e-7 + weighted_final_cam_image.max())
        image = torch.squeeze(image, dim = 0)

        return image, weighted_final_cam_image
        # return image, gray_image