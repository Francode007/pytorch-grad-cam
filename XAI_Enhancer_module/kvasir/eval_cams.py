"""
Kvasir-v2 CAM evaluation: compare Enhanced CAM vs standard methods using the same
insertion/deletion AUC and ROAD metrics as in enhanced_combiner/run_experiment.py.
Reuses ImageNetProperAUCEvaluator logic by subclassing and overriding data loading.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.kvasir.data import (
    load_split_file,
    KVASIR_CLASSES,
    IDX_TO_CLASS,
)
from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import (
    ImageNetProperAUCEvaluator,
    _ImageNetDataset,
)
from XAI_Enhancer_module.utils.model_utils import get_device


class KvasirProperAUCEvaluator(ImageNetProperAUCEvaluator):
    """
    Kvasir-v2 CAM evaluator. Reuses all metric code (insertion AUC, deletion AUC, ROAD)
    from ImageNetProperAUCEvaluator; only data source and model loading are Kvasir-specific.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        arch: str = "resnet50",
        split: str = "val",
        device_preference: str = "auto",
        layer_mode: str = "last",
        enhanced_cam_method: str = "GradCAMEnhanced",
        extractor_cls=None,
        extractor_kwargs: dict = None,
    ):
        # Don't call super().__init__; set up Kvasir model and same transform as ImageNet
        self.model_name = arch
        self.model_cache_dir = ""
        torch.backends.cudnn.benchmark = True
        dev = get_device(device_preference)
        self.device = torch.device(dev) if isinstance(dev, str) else dev
        self.imagenet_path = data_root  # used as data root for get_imagenet_images override
        self._kvasir_split = split
        self._kvasir_data_root = Path(data_root)
        self._checkpoint_path = checkpoint_path
        self.layer_mode = layer_mode
        self.enhanced_cam_method = enhanced_cam_method
        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs or {}
        self.synset_mapping = {str(i): c for i, c in enumerate(KVASIR_CLASSES)}  # dummy for any refs
        self.class_names = KVASIR_CLASSES

        # Same transform as ImageNet (Resize 256, CenterCrop 224, ImageNet norm)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load Kvasir model from checkpoint
        from XAI_Enhancer_module.kvasir.data import KVASIR_NUM_CLASSES
        model = build_kvasir_model(arch, num_classes=KVASIR_NUM_CLASSES, pretrained=False)
        load_kvasir_checkpoint(model, checkpoint_path, self.device)
        self.model = model.to(self.device)
        self.clean_model = model  # same model for metrics
        self.model.eval()
        self.clean_model.eval()

        self.conv_layers = self._get_enhanced_cam_layers(layer_mode)
        self.enhanced_cam_extractor = None
        print(f"KvasirProperAUCEvaluator: arch={arch}, split={split}, data_root={data_root}")

    def get_imagenet_images(self, max_images: int = -1, classes_filter=None, start_index: int = 0, end_index: int = None):
        """Override: return Kvasir val images (paths, predicted_labels, class_names)."""
        splits_dir = self._kvasir_data_root / "splits"
        split_file = splits_dir / f"{self._kvasir_split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}. Run prepare_splits first.")
        pairs = load_split_file(split_file, self._kvasir_data_root)
        all_image_paths = [str(p) for p, _ in pairs]
        all_labels = [lbl for _, lbl in pairs]
        all_class_names = [IDX_TO_CLASS[lbl] for lbl in all_labels]
        if end_index is not None:
            all_image_paths = all_image_paths[start_index:end_index]
            all_class_names = all_class_names[start_index:end_index]
            all_labels = all_labels[start_index:end_index]
        elif start_index > 0:
            all_image_paths = all_image_paths[start_index:]
            all_class_names = all_class_names[start_index:]
            all_labels = all_labels[start_index:]
        elif max_images > 0:
            all_image_paths = all_image_paths[:max_images]
            all_class_names = all_class_names[:max_images]
            all_labels = all_labels[:max_images]
        predicted_labels = self._predict_batch(all_image_paths)
        print(f"Collected {len(all_image_paths)} Kvasir {self._kvasir_split} images")
        return all_image_paths, predicted_labels, all_class_names


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Standard vs Enhanced CAM on Kvasir-v2 val set")
    p.add_argument("--data-root", type=str, default="data/kvasir-v2")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "resnet18", "resnet34", "densenet121"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to Kvasir-trained checkpoint")
    p.add_argument("--methods", type=str, default="gradcam,gradcampp,enhancedcam",
                  help="Comma-separated: gradcam, gradcampp, hirescam, scorecam, ablationcam, enhancedcam")
    p.add_argument("--enhanced-method", type=str, default="stagewise",
                  choices=["standard", "stagewise", "topk", "temp", "pyramid"], help="Aggregation for Enhanced CAM")
    p.add_argument("--layer-mode", type=str, default="last", choices=["last", "last_5", "all"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--step-size", type=int, default=224)
    p.add_argument("--max-images", type=int, default=-1, help="Cap number of val images (-1 = all)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="runs/kvasir/cam_eval")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base_cam_map = {
        "GradCAM": "GradCAMEnhanced",
        "GradCAM++": "GradCAMPlusPlusEnhanced",
        "HiResCAM": "HiResCAMEnhanced",
        "ScoreCAM": "ScoreCAMEnhanced",
        "AblationCAM": "AblationCAMEnhanced",
    }
    metrics_config = {"type": args.enhanced_method, "k": 5, "k_percent": 0.2, "temp": 0.05, "beta": 0.3, "soft": True}
    from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2

    methods = [m.strip().lower() for m in args.methods.split(",")]
    all_results = []
    compare_standard = "enhancedcam" in methods or any(m in methods for m in ["gradcam", "gradcam++", "hirescam", "scorecam", "ablationcam"])

    # Enhanced CAM (if requested)
    if "enhancedcam" in methods:
        evaluator = KvasirProperAUCEvaluator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            arch=args.arch,
            split=args.split,
            device_preference=args.device,
            layer_mode=args.layer_mode,
            enhanced_cam_method="GradCAMEnhanced",
            extractor_cls=EnhancedExtractorV2,
            extractor_kwargs={"aggregation_config": metrics_config},
        )
        res = evaluator.evaluate_enhanced_cam(
            max_images=args.max_images,
            step_size=args.step_size,
            verbose=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        all_results.append({
            "Method": f"EnhancedCAM ({args.enhanced_method})",
            "Insertion_Mean": res["insertion_auc_mean"],
            "Insertion_Std": res["insertion_auc_std"],
            "Deletion_Mean": res["deletion_auc_mean"],
            "Deletion_Std": res["deletion_auc_std"],
            "ROAD_Mean": res["road_mean"],
            "ROAD_Std": res["road_std"],
            "Images_Evaluated": res["num_images"],
        })
        print(f"EnhancedCAM: Ins={res['insertion_auc_mean']:.4f} Del={res['deletion_auc_mean']:.4f} ROAD={res['road_mean']:.4f}")

    # Standard methods
    standard_list = [m for m in methods if m in ["gradcam", "gradcam++", "hirescam", "scorecam", "ablationcam"] and m != "enhancedcam"]
    name_map = {"gradcam": "GradCAM", "gradcam++": "GradCAM++", "hirescam": "HiResCAM", "scorecam": "ScoreCAM", "ablationcam": "AblationCAM"}
    for m in standard_list:
        cam_name = name_map[m]
        std_cam = base_cam_map.get(cam_name, cam_name + "Enhanced")
        evaluator = KvasirProperAUCEvaluator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            arch=args.arch,
            split=args.split,
            device_preference=args.device,
            layer_mode="last",
            enhanced_cam_method=std_cam,
            extractor_cls=EnhancedExtractorV2,
            extractor_kwargs={"aggregation_config": {"type": "standard"}},
        )
        res = evaluator.evaluate_enhanced_cam(
            max_images=args.max_images,
            step_size=args.step_size,
            verbose=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        all_results.append({
            "Method": cam_name,
            "Insertion_Mean": res["insertion_auc_mean"],
            "Insertion_Std": res["insertion_auc_std"],
            "Deletion_Mean": res["deletion_auc_mean"],
            "Deletion_Std": res["deletion_auc_std"],
            "ROAD_Mean": res["road_mean"],
            "ROAD_Std": res["road_std"],
            "Images_Evaluated": res["num_images"],
        })
        print(f"{cam_name}: Ins={res['insertion_auc_mean']:.4f} Del={res['deletion_auc_mean']:.4f} ROAD={res['road_mean']:.4f}")

    df = pd.DataFrame(all_results)
    out_path = os.path.join(args.output_dir, "comparison_report.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
