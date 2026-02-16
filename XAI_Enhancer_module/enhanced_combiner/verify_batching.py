import torch
import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from XAI_Enhancer_module.utils.model_loader import ModelLoader

def verify_batching():
    print("--- Verifying Batch Processing Consistency ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Model
    root = Path(__file__).resolve().parent.parent.parent
    loader = ModelLoader(str(root / "XAI_Enhancer_module/pytorch_models"))
    model = loader.load_pretrained_model("resnet50").to(device)
    model.eval()
    
    # Init Extractor
    extractor = EnhancedExtractorV2(
        model, "resnet50", 
        conv_layers=[m for m in model.modules() if isinstance(m, torch.nn.Conv2d)],
        layer_batch_size=16
    )
    
    # Create Dummy Data
    B = 4
    torch.manual_seed(42)
    dummy_input = torch.randn(B, 3, 224, 224).to(device)
    labels = [0] * B
    
    # 1. Run Single (One by One)
    print("Running Single Mode...")
    single_outputs = []
    single_actuals = []
    
    # Warmup
    _ = extractor.extract_saliency_map(dummy_input[0].unsqueeze(0), 0)
    
    for i in range(B):
        img = dummy_input[i].unsqueeze(0)
        # Use underlying extractor logic to get intermediate
        img_ret, cam = extractor.extract_saliency_map(img, labels[i])
        single_outputs.append(cam.cpu()) # already normalized to [0,1]
        
        # Get actual output manually
        act = extractor.get_actual_output(img)
        single_actuals.append(act)
    
    single_stack = torch.stack(single_outputs)
    if single_stack.dim() == 4:
         single_stack = single_stack.squeeze(1)
         
    single_actuals_stack = np.stack(single_actuals)
    
    # 2. Run Batched
    print("Running Batched Mode...")
    _, batch_output = extractor.extract_saliency_map(dummy_input, labels)
    batch_output = batch_output.cpu()
    
    batch_actuals = extractor.get_actual_output_batch(dummy_input)
    
    # Compare Actuals
    diff_actuals = np.abs(single_actuals_stack - batch_actuals).max()
    print(f"Diff Actuals: {diff_actuals:.8f}")
    
    # Compare Final CAM
    print(f"Single Stack Shape: {single_stack.shape}")
    print(f"Batch Output Shape: {batch_output.shape}")
    
    diff = (single_stack - batch_output).abs().max().item()
    print(f"Max Difference CAM: {diff:.8f}")
    
    if diff_actuals > 1e-5:
        print("❌ Actual Output Mismatch!")
    else:
        print("✅ Actual Output Match!")
    
    if diff < 1e-4:
        print("✅ Batching Verification PASSED!")
    else:
        print("❌ Batching Verification FAILED!")

if __name__ == "__main__":
    verify_batching()
