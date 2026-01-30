
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import os

def prepare_real_data():
    target_count = 30
    base_dir = Path("imagenet_val_sample_test")
    base_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {target_count} images from ImageNet-1k (streaming)...")
    
    try:
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # fallback or exit
        return

    count = 0
    mapping_file = Path("test_mapping.txt")
    
    with open(mapping_file, 'w') as f_map:
        for sample in tqdm(ds, total=target_count):
            if count >= target_count:
                break
                
            img = sample['image']
            label = sample['label']
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # naming convention: val_{count}.JPEG
            filename = f"val_{count}.JPEG"
            save_path = base_dir / filename
            img.save(save_path)
            
            # Save label for test
            f_map.write(f"{filename} {label}\n")
            
            count += 1
            
    print(f"✅ Saved {count} images to {base_dir}")
    print(f"✅ Mapping saved to {mapping_file}")

if __name__ == "__main__":
    prepare_real_data()
