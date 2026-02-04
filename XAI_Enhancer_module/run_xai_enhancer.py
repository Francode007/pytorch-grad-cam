import argparse
import os
import subprocess
import sys
import getpass
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from download_models import download_all_models

def run_command(cmd_list):
    """Run a command using subprocess and stream output."""
    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

def create_imagenet_sample_from_hf(target_count=5000, base_path="imagenet_val_sample"):
    print(f"\nStarting download of {target_count} images from ImageNet-1k validation set...")
    
    # Load streaming dataset so we don't finish disk space
    try:
        print("Loading dataset from ILSVRC/imagenet-1k...")
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("did you accept the manual terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k ?")
        return None

    # --- Synset Mapping Handling ---
    mapping_path = None
    possible_paths = [
        "../LOC_synset_mapping.txt",
        "./LOC_synset_mapping.txt",
        "LOC_synset_mapping.txt"
    ]
    
    for p in possible_paths:
        if Path(p).exists():
            mapping_path = Path(p)
            print(f"Found synset mapping at: {mapping_path}")
            break

    # Download the official class index JSON for our own lookup
    urls = [
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
        "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json",
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    ]
    
    class_index = None
    for url in urls:
        try:
            print(f"Attempting to download class index from: {url}")
            with urllib.request.urlopen(url) as response:
                class_index = json.load(response)
            print("✅ Successfully downloaded class index.")
            break
        except Exception as e:
            print(f"⚠️ Failed to download from {url}: {e}")
            
    if class_index is None:
        print("❌ Could not download class index from any source.")
        return None

    if not mapping_path:
        mapping_path = Path("../LOC_synset_mapping.txt")
        print(f"⚠️ Synset mapping file missing. Generating it at {mapping_path}...")
        try:
            with open(mapping_path, 'w') as f:
                for idx in range(1000):
                    entry = class_index[str(idx)]
                    synset = entry[0]
                    name = entry[1]
                    f.write(f"{synset} {name}\n")
            print("✅ Created synset mapping file.")
        except Exception as e:
            print(f"Failed to write mapping file: {e}")
            print(f"Created synset mapping file at {mapping_path}")
    else:
         print(f"Synset mapping already exists at {mapping_path}")

    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    print("Streaming and saving images...")
    
    for sample in tqdm(ds, total=target_count):
        if count >= target_count:
            break
        
        img = sample['image']
        label_idx = sample['label']
        
        synset_id = class_index[str(label_idx)][0]
        
        synset_dir = base_dir / synset_id
        synset_dir.mkdir(exist_ok=True)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        save_path = synset_dir / f"val_{count}.JPEG"
        if not save_path.exists():
            img.save(save_path)
        
        count += 1
        
    print(f"\n✅ Successfully saved {count} images to {base_path}")
    return str(base_dir)

def main():
    parser = argparse.ArgumentParser(description='XAI Enhancer Runner')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., resnet50)')
    parser.add_argument('--base_cam', type=str, required=True, help='Base CAM method (e.g., HiResCAM)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Processing chunk size for restartability')
    parser.add_argument('--gpu-batch-size', type=int, default=1024, help='Batch size for GPU inference')
    parser.add_argument('--total-images', type=int, default=5000, help='Total images to evaluate')
    parser.add_argument('--step-size', type=int, default=224, help='Pixel step size')
    parser.add_argument('--dataset-path', type=str, default='imagenet_val_sample', help='Path to dataset')
    
    args = parser.parse_args()

    # --- Credentials ---
    print("\n--- Credential Setup ---")
    hf_token = os.environ.get("HF_TOKEN")
    manual_token = "" # Manual HF_TOKEN if needed
    
    if hf_token:
        print("Found HF_TOKEN in environment, logging in...")
        login(token=hf_token, add_to_git_credential=False)
    elif manual_token:
        print("Using manual HF token...")
        login(token=manual_token, add_to_git_credential=False)
    else:
        print("Please enter your Hugging Face Access Token (or press Enter if already logged in/not needed for cached data):")
        try:
             login(add_to_git_credential=False)
        except Exception as e:
             print(f"Login skipped/failed: {e}")

    email_to = "fnsaikia@gmail.com"
    email_sender = "fnsaikia@gmail.com"
    
    # Check if we should ask for password (only if we want email notifications)
    # The user asked to "keep the password as a manual entry"
    # We will provide a simple input prompts
    print(f"\nUsing default email recipient/sender: {email_to}")
    email_password = input("Enter Email Password (leave blank to skip email notifications): ").strip()
    
    # --- 1. Dataset Setup ---
    dataset_dir = args.dataset_path
    if not os.path.exists(dataset_dir) or len(list(Path(dataset_dir).rglob("*.JPEG"))) < args.total_images:
        print(f"\nDataset not found or incomplete at {dataset_dir}. Generating...")
        dataset_dir = create_imagenet_sample_from_hf(target_count=args.total_images, base_path=dataset_dir)
        if not dataset_dir:
            print("Failed to create dataset. Exiting.")
            sys.exit(1)
    else:
        print(f"\nDataset found at {dataset_dir}")

    # --- 2. Model Download ---
    print("\n--- Model Setup ---")
    MODEL_CACHE_DIR = "./pytorch_models"
    download_all_models(custom_folder=MODEL_CACHE_DIR)

    # --- 3. Batch Evaluation ---
    num_workers = str(os.cpu_count() or 4)
    enhanced_method = f"{args.base_cam}Enhanced"
    
    print(f"\n🚀 Starting Batch Evaluation: {args.total_images} images in chunks of {args.batch_size}")
    print(f"Model: {args.model}")
    print(f"Base CAM: {args.base_cam} (Enhanced: {enhanced_method})")
    print(f"GPU Batch Size: {args.gpu_batch_size}")
    
    # Phase 1: Enhanced CAM
    print(f"\n{'='*20} PHASE 1: ENHANCED CAM {'='*20}")
    for start_idx in range(0, args.total_images, args.batch_size):
        end_idx = min(start_idx + args.batch_size, args.total_images)
        print(f"\n▶️ Processing Chunk: {start_idx} to {end_idx}")
        
        cmd = [
            sys.executable, "imagenet_evaluation.py",
            "--model", args.model,
            "--imagenet-path", dataset_dir,
            "--eval-type", "enhanced-only",
            "--enhanced-cam-method", enhanced_method,
            "--model-cache-dir", MODEL_CACHE_DIR,
            "--device", "cuda",
            "--layer-mode", "all",
            "--start-index", str(start_idx),
            "--end-index", str(end_idx),
            "--batch-size", str(args.gpu_batch_size),
            "--step-size", str(args.step_size),
            "--save-intermediate",
            "--num-workers", num_workers
        ]
        
        if email_password:
            cmd.extend([
                "--email-to", email_to,
                "--email-sender", email_sender,
                "--email-password", email_password
            ])
            
        rc = run_command(cmd)
        if rc != 0:
            print(f"⚠️ Phase 1 chunk {start_idx}-{end_idx} failed with code {rc}")

    # Phase 2: Standard Methods
    print(f"\n{'='*20} PHASE 2: STANDARD METHODS {'='*20}")
    for start_idx in range(0, args.total_images, args.batch_size):
        end_idx = min(start_idx + args.batch_size, args.total_images)
        print(f"\n▶️ Processing Chunk: {start_idx} to {end_idx}")
        
        cmd = [
            sys.executable, "imagenet_evaluation.py",
            "--model", args.model,
            "--imagenet-path", dataset_dir,
            "--eval-type", "standard-only",
            "--methods", args.base_cam,
            "--model-cache-dir", MODEL_CACHE_DIR,
            "--device", "cuda",
            "--start-index", str(start_idx),
            "--end-index", str(end_idx),
            "--batch-size", str(args.gpu_batch_size),
            "--step-size", str(args.step_size),
            "--save-intermediate",
            "--num-workers", num_workers
        ]
        
        if email_password:
            cmd.extend([
                "--email-to", email_to,
                "--email-sender", email_sender,
                "--email-password", email_password
            ])
            
        rc = run_command(cmd)
        if rc != 0:
            print(f"⚠️ Phase 2 chunk {start_idx}-{end_idx} failed with code {rc}")

    # Phase 3: Aggregation
    print(f"\n{'='*20} PHASE 3: AGGREGATION {'='*20}")
    results_dir = f"./analysis_results/{args.model}_imagenet"
    
    agg_cmd = [
        sys.executable, "imagenet_evaluation.py",
        "--model", args.model,
        "--imagenet-path", dataset_dir,
        "--aggregate-dir", results_dir,
        "--output-csv-dir", "./csv_exports"
    ]
    
    if email_password:
        agg_cmd.extend([
            "--email-to", email_to,
            "--email-sender", email_sender,
            "--email-password", email_password
        ])
        
    run_command(agg_cmd)

if __name__ == "__main__":
    main()
