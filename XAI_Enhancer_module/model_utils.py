import os
import glob
import random
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import timm
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device configuration
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Paths and constants
BASE_MODEL_PATH = "/Users/f0s03xp/Desktop/IBS-research/models"
TRAIN_DATA_PATH = '/Users/f0s03xp/Desktop/IBS-research/og_data/IBS-preprocessed-dataset'
MEAN = [0.6380, 0.3422, 0.2275]
STD = [0.2448, 0.2060, 0.1710]
BATCH_SIZE = 16

# Class mappings
CLASSES = ['IBS', 'Normal']
CLASSES.sort()
IDX_TO_CLASS = dict(enumerate(CLASSES))
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

# Image transformations
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Dataset
class IBSValDataset(Dataset):
    def __init__(self, image_paths, model_name):
        self.image_paths = image_paths
        self.model_name = model_name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = plt.imread(image_filepath)
        img_size = 384 if self.model_name.startswith("b4") else 224
        image = cv2.resize(image, (img_size, img_size))
        image = image.astype(float) / 255.0
        image = transformations(image).float()
        image_name = os.path.splitext(os.path.basename(image_filepath))[0]
        return image, image_name, image_filepath

# Data loader
def get_validation_paths(data_path):
    image_paths = []
    for class_dir in glob.glob(os.path.join(data_path, '*')):
        for img_path in glob.glob(os.path.join(class_dir, '*')):
            image_paths.append(img_path)
    random.shuffle(image_paths)
    split_idx = int(0.8 * len(image_paths))
    return image_paths[split_idx:]

def get_val_dataloader(model_name):
    valid_image_paths = get_validation_paths(TRAIN_DATA_PATH)
    dataset = IBSValDataset(valid_image_paths, model_name)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model builder
def build_model_inf(model_name, num_classes=2, base_model_path=BASE_MODEL_PATH):
    if model_name == 'b0':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(1280, num_classes)
        model_path = f"{base_model_path}/effnetb0.pth"
    elif model_name == 'b4':
        model = models.efficientnet_b4(pretrained=False)
        model.classifier[1] = nn.Linear(1792, num_classes)
        model_path = f"{base_model_path}/effnetb4.pth"
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        model_path = f"{base_model_path}/resnet50.pth"
    elif model_name == 'densenet':
        model = timm.create_model('densenet121', pretrained=False)
        model.classifier = nn.Linear(1024, num_classes)
        model_path = f"{base_model_path}/densenet121.pth"
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        model_path = f"{base_model_path}/resnet18.pth"
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        model_path = f"{base_model_path}/resnet34.pth"
    elif model_name == 'xception':
        model = timm.create_model('xception', pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        model_path = f"{base_model_path}/xception_net.pth"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model, model_path

def pred_model(model_name, num_classes=2, base_model_path=BASE_MODEL_PATH):
    device = get_device()
    model, model_path = build_model_inf(model_name, num_classes, base_model_path)
    # Fix for PyTorch 2.6 weights_only default change
    try:
        state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    except Exception:
        # Fallback to weights_only=False for older model files
        state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    return model

def test_model(model_name, num_classes=2, base_model_path=BASE_MODEL_PATH):
    device = get_device()
    model, model_path = build_model_inf(model_name, num_classes, base_model_path)
    # Fix for PyTorch 2.6 weights_only default change
    try:
        state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    except Exception:
        # Fallback to weights_only=False for older model files
        state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

# Inference
def inference(model, testloader, model_name):
    print(f"Predicting labels for model: {model_name}")
    predictions, image_filepaths, image_names = [], [], []
    device = get_device()
    with torch.no_grad():
        for images, names, filepaths in tqdm(testloader, total=len(testloader)):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            preds = [IDX_TO_CLASS[p.item()] for p in preds]
            predictions.extend(preds)
            image_filepaths.extend(filepaths)
            image_names.extend(names)
    return predictions, image_filepaths, image_names

# Example usage (uncomment to run)
# model_name = 'resnet50'
# model = pred_model(model_name)
# val_loader = get_val_dataloader(model_name)
# preds, paths, names = inference(model, val_loader, model_name)