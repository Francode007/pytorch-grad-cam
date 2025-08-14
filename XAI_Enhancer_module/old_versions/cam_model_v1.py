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
from XAI_Enhancer_module.utils.model_utils import get_device, transformations, CLASS_TO_IDX, IDX_TO_CLASS

# Updated to use device preference
def get_device_for_cam(device_preference="auto"):
    return get_device(device_preference)

device = get_device_for_cam()

class CamDataset(Dataset):
    def __init__(self, cam, model_name, image_filepaths, labels, model, conv_list):
        self.cam = cam
        self.model_name = model_name
        self.image_filepaths = image_filepaths
        self.labels = labels
        self.tfms = transformations
        self.model = model
        self.conv_list = conv_list

    def __len__(self):
        return len(self.image_filepaths)
    
    def modified_cam(self, image, label, target_layers):
        
        cam = self.cam(self.model, target_layers)
        targets = [ClassifierOutputTarget(label)]
        cam_per_layer, mod_outputs_per_layer = cam(input_tensor=image, targets=targets)
        # grayscale_cam = grayscale_cam[0, :] implemented in the for loop
        return image, cam_per_layer, mod_outputs_per_layer
    
    def class_to_label(self, label):
        return CLASS_TO_IDX[label]
    
    def hook_and_cook(self, target_layer, model, input_tensor, intermediate_output):
        def hook(module, input, output):
            output = intermediate_output
            return output
            
        # hook_1 = target_layer[0].register_forward_hook(hook)
        hook_1 = target_layer.register_forward_hook(hook)

        modified_output = model(input_tensor)
        modified_output = modified_output[0].cpu().data.numpy()
        hook_1.remove()

        return modified_output
    
    def forward_pass_actual(self, input):
        actual_output = self.model(input)
        actual_output = actual_output[0].cpu().data.numpy()
        return actual_output
    
    def transform_image(self, image):
        img_size = [384 if self.model_name.startswith("b4") else 224]
        image = cv2.resize(image, (img_size[0], img_size[0]))
        image = image/255.0
        image = self.tfms(image)
        image = image.float()
        input_tensor = torch.unsqueeze(image, dim = 0)
        return input_tensor
    
    def cosine_similarity_(self, actual_output, modified_output):
        """
        Calculates the cosine similarity between actual_output and modified_output vectors.
        Returns the similarity as a numpy array.
        """
        actual = torch.tensor(actual_output) if not isinstance(actual_output, torch.Tensor) else actual_output
        modified = torch.tensor(modified_output) if not isinstance(modified_output, torch.Tensor) else modified_output
        # Ensure both are 2D (batch, features)
        if actual.ndim == 1:
            actual = actual.unsqueeze(0)
        if modified.ndim == 1:
            modified = modified.unsqueeze(0)
        cos = nn.functional.cosine_similarity(actual, modified, dim=1)
        return cos.detach().cpu().numpy()

    def __getitem__(self, idx):
        '''
        Implementation with softmax weighing scheme, 
        after applying cosine_similarity_
        '''
        image_filepath = self.image_filepaths[idx]
        image = plt.imread(image_filepath)
        label = self.labels[idx]
        pred_label = self.class_to_label(label)
        input_tensor = self.transform_image(image)
        actual_output = self.forward_pass_actual(input_tensor)
        image, cam_per_layer, modified_output_per_layer = self.modified_cam(image, pred_label, self.conv_list)
        cos_values = []
        for index, layer in enumerate(self.conv_list):
            intermediate_output = modified_output_per_layer[index]
            intermediate_map = torch.from_numpy(intermediate_output)
            modified_output = self.hook_and_cook(layer, self.model, image, intermediate_map)
            cos_val = self.cosine_similarity_(actual_output, modified_output)
            cos_values.append(cos_val)

        # Convert cos_values to a tensor and apply softmax
        cos_values_tensor = torch.tensor(cos_values, dtype=torch.float32).squeeze()
        softmax_weights = torch.softmax(cos_values_tensor, dim=0)

        weighted_final_cam_image = 0
        for i, weight in enumerate(softmax_weights):
            cam_image = torch.from_numpy(cam_per_layer[i][0, :])
            weighted_final_cam_image += weight * cam_image

            weighted_final_cam_image -= weighted_final_cam_image.min()
            weighted_final_cam_image /= (1e-7 + weighted_final_cam_image.max())
            image = torch.squeeze(image, dim = 0)

        return image, weighted_final_cam_image
        # return image, gray_image