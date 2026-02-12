
import os
import sys
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

def create_imagenet_sample_from_hf(target_count=50, base_path="imagenet_val_sample"):
    print(f"\nStarting download of {target_count} images from ImageNet-1k validation set...")
    
    # Try to login if token is in env
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Found HF_TOKEN, logging in...")
        login(token=hf_token, add_to_git_credential=False)
    
    # Load streaming dataset so we don't finish disk space
    try:
        print("Loading dataset from ILSVRC/imagenet-1k...")
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("did you accept the manual terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k ?")
        return None

    # --- Synset Mapping Handling ---
    mapping_path = Path("LOC_synset_mapping.txt")
    
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

    if not mapping_path.exists():
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

if __name__ == "__main__":
    create_imagenet_sample_from_hf()
