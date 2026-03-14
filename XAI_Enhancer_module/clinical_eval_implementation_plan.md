# XAI-Enhancer: Clinical Alignment via Expert Mask Evaluation

## Objective
To objectively evaluate the clinical faithfulness of the XAI-Enhancer by quantifying the spatial overlap between the generated classification heatmaps and human-expert ground-truth segmentation masks. This serves as a mathematical proxy for a human-in-the-loop clinical study.

## Prerequisites
1. **Dataset:** Download the **Kvasir-SEG** dataset (which contains original images and corresponding expert binary masks).
2. **Model:** Your existing pre-trained PyTorch classification model (e.g., ResNet-50 trained on Kvasir-v2).
3. **Libraries:** `torch`, `torchvision`, `numpy`, `cv2` (OpenCV), `sklearn`, `matplotlib`.

## Implementation Steps

### Phase 1: Data Loading & Preprocessing
* Create a custom PyTorch `Dataset` class that loads a Kvasir-SEG image and its corresponding ground-truth binary mask simultaneously.
* Apply the exact same resizing and normalization transforms to the input image as used during your original training.
* Resize the ground-truth mask to match the network's input dimensions (e.g., $224 \times 224$).

### Phase 2: Heatmap Generation & Binarization
* Pass the input image through the network to get the target class prediction.
* Generate the continuous saliency heatmaps (values between 0 and 1) using both the **Base CAM (e.g., Grad-CAM)** and the **XAI-Enhancer**.
* **Binarization:** Because IoU and Dice require binary inputs, apply a threshold to the heatmaps to convert them into binary masks (e.g., setting all pixels with an activation $> 0.5$ to $1$, and the rest to $0$). Otsu's thresholding can also be used for dynamic binarization.

### Phase 3: Metric Calculation (IoU and Dice)
For each image, calculate:
1. **Intersection over Union (IoU):** $\frac{\text{Area of Overlap}}{\text{Area of Union}}$ between the binarized heatmap and the expert mask.
2. **Dice Coefficient:** $\frac{2 \times \text{Area of Overlap}}{\text{Total Pixels in Heatmap} + \text{Total Pixels in Expert Mask}}$.

### Phase 4: Aggregation and Visualization
* Compute the Mean IoU (mIoU) and Mean Dice across the entire Kvasir-SEG dataset for both methods.
* Generate a Matplotlib grid saving visual examples where XAI-Enhancer successfully bounded the pathology while Grad-CAM suffered from spatial diffusion.