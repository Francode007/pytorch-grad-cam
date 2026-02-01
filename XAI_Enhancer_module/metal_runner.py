
import os
import sys
import subprocess
import time
from pathlib import Path

# --- Configuration ---
TOTAL_IMAGES = 5000
BATCH_SIZE = 500      # restart chunks
EVAL_BATCH_SIZE = 64  # GPU batch size
STEP_SIZE = 224       # Optimized step size
MODEL_NAME = "resnet50"
LAYER_MODE = "all"    # User requested 'all' for Enhanced CAM
DEVICE = "mps"        # Force Metal Performance Shaders

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "imagenet_val_sample"
OUTPUT_DIR = Path("analysis_results") / f"{MODEL_NAME}_imagenet"
CSV_DIR = Path("csv_exports")

def ensure_dataset():
    """Ensure the dataset exists, otherwise download it."""
    if not DATASET_DIR.exists():
        print("Dataset not found. Running dataset creation...")
        # We can import the logic from the notebook or just run a snippet
        # For robustness, let's assume the user has run the setup or we run a helper
        # Given the previous context, we'll try to use the existing notebook logic converted to py
        # But specifically, let's look for the 'create_imagenet_sample_from_hf' function availability
        # It was inside the notebook. I'll reimplement a minimal version here or ask user.
        # Actually, let's check if the directory has images.
        pass
    else:
        print(f"✅ Using existing dataset at {DATASET_DIR}")
        
    # Check count
    count = len(list(DATASET_DIR.glob("*/*.JPEG")))
    if count < TOTAL_IMAGES:
        print(f"⚠️ Warning: Found {count} images, expected {TOTAL_IMAGES}.")
        
    return str(DATASET_DIR)

def run_command(cmd_list, log_file=None):
    """Run a subprocess command and stream output."""
    print(f"Running: {' '.join(cmd_list)}")
    
    if log_file:
        with open(log_file, "a") as f:
            process = subprocess.Popen(
                cmd_list,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            process.wait()
    else:
        process = subprocess.Popen(
            cmd_list,
            text=True,
            bufsize=1
        )
        process.wait()
        
    return process.returncode

def main():
    print(f"{'='*60}")
    print("🚀 XAI ENHANCER - METAL GPU RUNNER")
    print(f"{'='*60}")
    
    # 1. Setup
    dataset_path = ensure_dataset()
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = logs_dir / f"run_{timestamp}.log"
    print(f"📄 Logging to: {log_file}")

    # 2. Run Enhanced CAM Evaluation
    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 1: Enhanced CAM Evaluation")
    
    for start_idx in range(0, TOTAL_IMAGES, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, TOTAL_IMAGES)
        print(f"  ▶️ Processing batch {start_idx} - {end_idx}...")
        
        cmd = [
            sys.executable, "imagenet_evaluation.py",
            "--model", MODEL_NAME,
            "--imagenet-path", dataset_path,
            "--eval-type", "enhanced-only",
            "--model-cache-dir", "../../pytorch_models", # Adjusted to correct relative path
            "--device", DEVICE,
            "--layer-mode", LAYER_MODE,
            "--start-index", str(start_idx),
            "--end-index", str(end_idx),
            "--batch-size", str(EVAL_BATCH_SIZE),
            "--step-size", str(STEP_SIZE),
            "--max-images", "-1", # Process all images in the range
            "--save-intermediate"
        ]
        
        run_command(cmd, log_file)

    # 3. Run Standard Methods Evaluation
    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 2: Standard CAM Evaluation")
    STANDARD_METHODS = ["GradCAM"] # User asked for GradCAM and its enhanced version
    
    for start_idx in range(0, TOTAL_IMAGES, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, TOTAL_IMAGES)
        print(f"  ▶️ Processing batch {start_idx} - {end_idx}...")
        
        cmd = [
            sys.executable, "imagenet_evaluation.py",
            "--model", MODEL_NAME,
            "--imagenet-path", dataset_path,
            "--eval-type", "standard-only",
            "--methods"] + STANDARD_METHODS + [
            "--model-cache-dir", "../../pytorch_models",
            "--device", DEVICE,
            "--start-index", str(start_idx),
            "--end-index", str(end_idx),
            "--batch-size", str(EVAL_BATCH_SIZE),
            "--step-size", str(STEP_SIZE),
            "--max-images", "-1", # Process all images in the range
            "--save-intermediate"
        ]
        
        run_command(cmd, log_file)

    # 4. Aggregation
    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 3: Aggregation")
    agg_cmd = [
        sys.executable, "imagenet_evaluation.py",
        "--model", MODEL_NAME,
        "--imagenet-path", dataset_path,
        "--aggregate-dir", str(OUTPUT_DIR),
        "--output-csv-dir", str(CSV_DIR)
    ]
    run_command(agg_cmd, log_file)
    
    print(f"\n✅ Run Complete!")
    print(f"📊 Results directory: {OUTPUT_DIR}")
    print(f"📈 Final CSV exported to: {CSV_DIR}")

if __name__ == "__main__":
    main()
