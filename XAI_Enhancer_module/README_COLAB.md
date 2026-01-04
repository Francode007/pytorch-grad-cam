# Running XAI Enhancer on Google Colab

This guide explains how to run the XAI Enhancer module on Google Colab using the included `XAI_Enhancer_Colab_Runner.ipynb` notebook.

## Prerequisites

-   A Google account to access [Google Colab](https://colab.research.google.com/).
-   **Important**: A [Hugging Face](https://huggingface.co/) account and Access Token.
    -   You must accept the access conditions for the **ImageNet-1k** dataset here: [https://huggingface.co/datasets/imagenet-1k](https://huggingface.co/datasets/imagenet-1k).
    -   Create an Access Token in your Hugging Face settings (read access is sufficient).

## Steps

1.  **Open Google Colab**: Go to [https://colab.research.google.com/](https://colab.research.google.com/).

2.  **Upload Notebook**:
    *   Click on the **Upload** tab.
    *   Drag and drop the `XAI_Enhancer_Colab_Runner.ipynb` file from this folder.

3.  **Upload Module Files**:
    *   Once the notebook is open, click the **Folder icon** 📁 on the left sidebar.
    *   **Recommended**: Zip the entire `XAI_Enhancer_module` folder on your computer, upload it, and unzip it in Colab (`!unzip XAI_Enhancer_module.zip`).
    *   Ensure all scripts (`imagenet_evaluation.py`, `utils/`, etc.) AND `LOC_synset_mapping.txt` (if available in parent dir) are accessible.

4.  **Run the Notebook**:
    *   **Runtime**: It is highly recommended to use a **GPU Runtime** (Runtime -> Change runtime type -> T4 GPU) for evaluating 5000 images.
    *   **Step 1**: Installs dependencies.
    *   **Step 2**: Authenticates with Hugging Face (paste your token) and streams **5000 images** from the official ImageNet validation set.
    *   **Step 3**: Downloads pre-trained models.
    *   **Step 4**: Runs the evaluation.

## Troubleshooting

-   **"Error loading dataset"**: Did you accept the terms on the Hugging Face ImageNet-1k page? Did you enter a valid token?
-   **Model download fails**: Check your internet connection.
-   **Module not found**: Ensure you are in the correct directory.

Enjoy exploring the XAI Enhancer!
