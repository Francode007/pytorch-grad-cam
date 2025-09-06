import os
import torch
import torchvision.models as models
from tqdm import tqdm

def download_all_models(custom_folder: str):
    """
    Downloads a predefined list of torchvision models to a custom folder.
    
    Args:
        custom_folder (str): The path to the directory where models will be saved.
    """
    # 1. Define the custom folder and set the TORCH_HOME environment variable
    # This tells PyTorch where to save cached files, including downloaded models.
    print(f"Setting model download directory to: {custom_folder}")
    os.environ['TORCH_HOME'] = custom_folder
    
    # Create the directory if it doesn't exist
    os.makedirs(custom_folder, exist_ok=True)

    # 2. List all the model names you want to download
    model_names = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'vgg16', 'vgg19',
        'densenet121', 'densenet169', 'densenet201',
        'mobilenet_v2', 'mobilenet_v3_large',
        'efficientnet_b0', 'efficientnet_b4'
    ]

    print("\nStarting model downloads...")
    # 3. Loop through the list and download each model
    for model_name in tqdm(model_names, desc="Downloading Models"):
        try:
            # Get the model-loading function from torchvision.models dynamically
            model_loader = getattr(models, model_name)
            
            # Load the model with the recommended 'weights' parameter
            # This is the modern replacement for 'pretrained=True'
            model_loader(weights='IMAGENET1K_V1')
            
            # No need to assign it to a variable, the download happens on this call
            
        except AttributeError:
            tqdm.write(f"⚠️ Model '{model_name}' not found in torchvision.models. Skipping.")
        except Exception as e:
            tqdm.write(f"❌ Failed to download '{model_name}'. Error: {e}")
            
    print("\n✅ All specified models have been downloaded successfully.")
    print(f"Check the 'hub/checkpoints' subdirectory inside: {custom_folder}")


if __name__ == '__main__':
    # --- Specify your custom folder here ---
    # You can use an absolute or relative path
    DOWNLOAD_DIRECTORY = "../../pytorch_models" 
    
    download_all_models(custom_folder=DOWNLOAD_DIRECTORY)