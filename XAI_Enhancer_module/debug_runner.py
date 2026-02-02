#!/usr/bin/env python3
import subprocess
import sys
import os

def run_command(cmd_list):
    print(f"Running command: {' '.join(cmd_list)}")
    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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

def main():
    # Configuration matches the notebook
    dataset_path = "imagenet_val_sample"
    MODEL_NAME = "resnet50"
    EVAL_BATCH_SIZE = 512
    STEP_SIZE = 224
    
    # Test a chunk that was failing (e.g. 300-600)
    start_idx = 300
    end_idx = 600
    
    print(f"Testing chunk {start_idx} to {end_idx}")

    cmd = [
        sys.executable, "imagenet_evaluation.py",
        "--model", MODEL_NAME,
        "--imagenet-path", dataset_path,
        "--eval-type", "enhanced-only",
        "--enhanced-cam-method", "HiResCAMEnhanced",
        "--model-cache-dir", "./pytorch_models",
        "--device", "auto",
        "--layer-mode", "all",
        "--start-index", str(start_idx),
        "--end-index", str(end_idx),
        "--batch-size", str(EVAL_BATCH_SIZE),
        "--step-size", str(STEP_SIZE),
        "--save-intermediate",
        "--num-workers", str(os.cpu_count())
    ]
    
    run_command(cmd)

if __name__ == "__main__":
    main()
