"""
IBS CAM evaluation: compare Enhanced CAM vs standard methods using the same
insertion/deletion AUC and ROAD metrics as in enhanced_combiner/run_experiment.py.
Reuses ImageNetProperAUCEvaluator logic by subclassing and overriding data loading.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.ibs.data import (
    load_split_file,
    IBS_CLASSES,
    IDX_TO_CLASS,
    IBS_MEAN,
    IBS_STD,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import (
    ImageNetProperAUCEvaluator,
    _ImageNetDataset,
)
from XAI_Enhancer_module.utils.model_utils import get_device
from XAI_Enhancer_module.utils.layercam_utils import (
    get_layercam_stage_layers,
    extract_layercam_fused,
)
from XAI_Enhancer_module.utils.hrcam_utils import (
    train_hrcam_head,
    load_hrcam_head,
    extract_hrcam,
)
from XAI_Enhancer_module.common.layer_sets import LAYER_SET_CHOICES
from XAI_Enhancer_module.common.cam_eval_protocol import (
    enforce_eval_split,
    write_protocol_header,
    per_image_stem,
)


class IBSProperAUCEvaluator(ImageNetProperAUCEvaluator):
    """
    IBS CAM evaluator. Reuses all metric code (insertion AUC, deletion AUC, ROAD)
    from ImageNetProperAUCEvaluator; only data source and model loading are IBS-specific.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        arch: str = "resnet50",
        split: str = "test",
        fold: int = None,
        device_preference: str = "auto",
        layer_mode: str = "last",
        enhanced_cam_method: str = "GradCAMEnhanced",
        extractor_cls=None,
        extractor_kwargs: dict = None,
    ):
        self.model_name = arch
        self.model_cache_dir = ""
        torch.backends.cudnn.benchmark = True
        dev = get_device(device_preference)
        self.device = torch.device(dev) if isinstance(dev, str) else dev
        self.imagenet_path = data_root  # used as data root for get_imagenet_images override
        self._ibs_split = split
        self._ibs_data_root = Path(data_root)
        self._ibs_fold = fold
        self._checkpoint_path = checkpoint_path
        self.layer_mode = layer_mode
        self.enhanced_cam_method = enhanced_cam_method
        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs or {}
        self.synset_mapping = {str(i): c for i, c in enumerate(IBS_CLASSES)}
        self.class_names = IBS_CLASSES

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IBS_MEAN, std=IBS_STD),
        ])

        from XAI_Enhancer_module.ibs.data import IBS_NUM_CLASSES
        model = build_ibs_model(arch, num_classes=IBS_NUM_CLASSES, pretrained=False)
        load_ibs_checkpoint(model, checkpoint_path, self.device)
        self.model = model.to(self.device)
        self.clean_model = model  # same model for metrics
        self.model.eval()
        self.clean_model.eval()

        self.conv_layers = self._get_enhanced_cam_layers(layer_mode)
        self.enhanced_cam_extractor = None
        print(f"IBSProperAUCEvaluator: arch={arch}, fold={fold}, split={split}, data_root={data_root}")

    def get_imagenet_images(self, max_images: int = -1, classes_filter=None, start_index: int = 0, end_index: int = None):
        """Override: return IBS fold split images (paths, predicted_labels, class_names)."""
        splits_dir = self._ibs_data_root / "splits"
        if self._ibs_fold is not None:
            splits_dir = splits_dir / f"fold{self._ibs_fold}"
        split_file = splits_dir / f"{self._ibs_split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. Run prepare-ibs-folds first "
                f"(expected splits/fold{{k}}/{{split}}.txt)."
            )
        pairs = load_split_file(split_file, self._ibs_data_root)
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
        print(f"Collected {len(all_image_paths)} IBS fold={self._ibs_fold} {self._ibs_split} images")
        return all_image_paths, predicted_labels, all_class_names


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Standard vs Enhanced CAM on IBS (test fold by default)")
    p.add_argument("--data-root", type=str, default="data/IBS-patient-dataset")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--allow-val",
        action="store_true",
        help="Permit --split val (forbidden by default; D-M6).",
    )
    p.add_argument("--fold", type=int, required=True, help="Patient fold index (0..n_folds-1)")
    p.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to IBS-trained checkpoint")
    p.add_argument(
        "--methods",
        type=str,
        default="gradcam,gradcampp,enhancedcam",
        help="Comma-separated: gradcam, gradcampp, hirescam, scorecam, ablationcam, "
             "enhancedcam, uniform, layercam, layercam_fused, hrcam",
    )
    p.add_argument(
        "--base-cam",
        type=str,
        default="GradCAM",
        choices=["GradCAM", "GradCAM++", "HiResCAM", "ScoreCAM", "AblationCAM"],
        help="Base CAM method to use for the enhanced variant (maps to corresponding *Enhanced class).",
    )
    p.add_argument("--gamma", type=float, default=2.0,
                    help="LayerCAM fusion: tanh scaling factor for shallow layers (paper Eq. 9, default 2)")
    p.add_argument("--fuse-stages", type=str, default=None,
                    help="LayerCAM fusion: comma-separated 1-based stage indices to fuse "
                         "(e.g. '2,3,4' for ResNet). Default: all stages except stage 1 for VGG, all for ResNet/DenseNet)")
    p.add_argument("--hrcam-checkpoint", type=str, default=None,
                    help="Path to pre-trained HR-CAM head checkpoint. If omitted, the head is "
                         "auto-trained on the training split before evaluation.")
    p.add_argument("--hrcam-epochs", type=int, default=20,
                    help="Epochs for HR-CAM head auto-training (default 20)")
    p.add_argument("--hrcam-lr", type=float, default=3e-4,
                    help="Learning rate for HR-CAM head auto-training (default 3e-4)")
    p.add_argument(
        "--enhanced-method",
        type=str,
        default="standard",
        choices=["standard", "uniform", "stagewise", "topk", "temp", "pyramid"],
        help="Aggregation for Enhanced CAM (default: standard / flat softmax, D-M2)",
    )
    p.add_argument(
        "--layer-set",
        "--layer-mode",
        dest="layer_set",
        type=str,
        default="all",
        choices=list(LAYER_SET_CHOICES),
        help="Target layer set for Enhanced/Uniform CAM (default: all)",
    )
    p.add_argument("--layer-batch-size", type=int, default=32, help="Layers processed in parallel (lower=less VRAM with layer-mode all)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for CAM eval (lower=less RAM/VRAM)")
    p.add_argument(
        "--step-size",
        type=int,
        default=224,
        help="Pixels per Insertion/Deletion step (224 → one row of a 224² image per step)",
    )
    p.add_argument("--road-seed", type=int, default=0, help="Seed recorded / applied before ROAD imputation")
    p.add_argument("--road-imputation", type=str, default="blur", choices=["blur", "black"])
    p.add_argument("--max-images", type=int, default=-1, help="Cap number of images (-1 = all)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="runs/ibs/cam_eval")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--compare-standard",
        action="store_true",
        help="Also evaluate standard CAMs (GradCAM, GradCAM++, HiResCAM, ScoreCAM, AblationCAM).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.split = enforce_eval_split(args.split, allow_val=args.allow_val)
    os.makedirs(args.output_dir, exist_ok=True)
    # Backward-compat alias used elsewhere in this file
    args.layer_mode = args.layer_set

    write_protocol_header(
        args.output_dir,
        {
            "dataset": "ibs",
            "fold": args.fold,
            "split": args.split,
            "arch": args.arch,
            "checkpoint": args.checkpoint,
            "methods": args.methods,
            "base_cam": args.base_cam,
            "enhanced_method": args.enhanced_method,
            "layer_set": args.layer_set,
            "step_size": args.step_size,
            "road_seed": args.road_seed,
            "road_imputation": args.road_imputation,
            "max_images": args.max_images,
            "batch_size": args.batch_size,
            "layer_batch_size": args.layer_batch_size,
        },
    )

    base_cam_map = {
        "GradCAM": "GradCAMEnhanced",
        "GradCAM++": "GradCAMPlusPlusEnhanced",
        "HiResCAM": "HiResCAMEnhanced",
        "ScoreCAM": "ScoreCAMEnhanced",
        "AblationCAM": "AblationCAMEnhanced",
    }
    metrics_config = {
        "type": args.enhanced_method,
        "k": 5,
        "k_percent": 0.2,
        "temp": 0.05,
        "beta": 0.3,
        "soft": True,
    }
    extractor_kwargs_base = {"layer_batch_size": args.layer_batch_size}
    from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2

    methods = [m.strip().lower() for m in args.methods.split(",")]
    # Treat "uniform" as enhancedcam with type=uniform
    if "uniform" in methods and "enhancedcam" not in methods:
        methods.append("enhancedcam")
        if args.enhanced_method == "standard":
            args.enhanced_method = "uniform"
            metrics_config["type"] = "uniform"

    all_results = []
    # Determine whether to run standard methods
    has_standard_in_methods = any(
        m in methods for m in ["gradcam", "gradcam++", "hirescam", "scorecam", "ablationcam"]
    )
    run_standard = args.compare_standard or has_standard_in_methods

    # Multi-layer aggregators need more than the last layer
    enhanced_layer_mode = args.layer_mode
    if (
        args.enhanced_method in ("standard", "uniform", "stagewise", "topk", "temp", "pyramid")
        and enhanced_layer_mode == "last"
        and ("enhancedcam" in methods or "uniform" in methods)
    ):
        print(
            f"INFO: Enhanced method '{args.enhanced_method}' needs multiple layers. "
            "Switching layer-set from 'last' to 'all' for Enhanced/Uniform CAM."
        )
        enhanced_layer_mode = "all"

    eval_kw = dict(
        step_size=args.step_size,
        verbose=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        road_seed=args.road_seed,
        road_imputation=args.road_imputation,
    )

    # Enhanced CAM (if requested)
    if "enhancedcam" in methods or "uniform" in methods:
        # Map chosen base CAM to its enhanced implementation
        enhanced_cam_name = base_cam_map[args.base_cam]
        method_label = (
            "Uniform (T→∞)"
            if metrics_config["type"] == "uniform"
            else f"EnhancedCAM ({args.base_cam})"
        )
        evaluator = IBSProperAUCEvaluator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            arch=args.arch,
            split=args.split,
            fold=args.fold,
            device_preference=args.device,
            layer_mode=enhanced_layer_mode,
            enhanced_cam_method=enhanced_cam_name,
            extractor_cls=EnhancedExtractorV2,
            extractor_kwargs={**extractor_kwargs_base, "aggregation_config": metrics_config},
        )
        res = evaluator.evaluate_enhanced_cam(
            max_images=args.max_images,
            method_name=method_label,
            per_image_path=per_image_stem(args.output_dir, method_label),
            **eval_kw,
        )
        all_results.append({
            "Method": method_label,
            "Insertion_Mean": res["insertion_auc_mean"],
            "Insertion_Std": res["insertion_auc_std"],
            "Deletion_Mean": res["deletion_auc_mean"],
            "Deletion_Std": res["deletion_auc_std"],
            "ROAD_Mean": res["road_mean"],
            "ROAD_Std": res["road_std"],
            "Images_Evaluated": res["num_images"],
        })
        print(f"{method_label}: Ins={res['insertion_auc_mean']:.4f} Del={res['deletion_auc_mean']:.4f} ROAD={res['road_mean']:.4f}")

    # Standard methods (single last layer only)
    standard_list = [
        m
        for m in methods
        if m in ["gradcam", "gradcam++", "hirescam", "scorecam", "ablationcam"] and m != "enhancedcam"
    ]
    if not run_standard:
        standard_list = []
    elif run_standard and not standard_list:
        standard_list = ["gradcam", "gradcam++", "hirescam"]
    name_map = {"gradcam": "GradCAM", "gradcam++": "GradCAM++", "hirescam": "HiResCAM", "scorecam": "ScoreCAM", "ablationcam": "AblationCAM"}
    for m in standard_list:
        cam_name = name_map[m]
        std_cam = base_cam_map.get(cam_name, cam_name + "Enhanced")
        evaluator = IBSProperAUCEvaluator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            arch=args.arch,
            split=args.split,
            fold=args.fold,
            device_preference=args.device,
            layer_mode="last",
            enhanced_cam_method=std_cam,
            extractor_cls=EnhancedExtractorV2,
            extractor_kwargs={**extractor_kwargs_base, "aggregation_config": {"type": "standard"}},
        )
        res = evaluator.evaluate_enhanced_cam(
            max_images=args.max_images,
            method_name=cam_name,
            per_image_path=per_image_stem(args.output_dir, cam_name),
            **eval_kw,
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

    # ------------------------------------------------------------------
    # LayerCAM (single-layer, last conv) and LayerCAM-Fused (multi-layer)
    # ------------------------------------------------------------------
    run_layercam = "layercam" in methods
    run_layercam_fused = "layercam_fused" in methods

    if run_layercam or run_layercam_fused:
        evaluator = IBSProperAUCEvaluator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            arch=args.arch,
            split=args.split,
            fold=args.fold,
            device_preference=args.device,
            layer_mode="last",
            enhanced_cam_method="GradCAMEnhanced",
            extractor_cls=EnhancedExtractorV2,
            extractor_kwargs={**extractor_kwargs_base, "aggregation_config": {"type": "standard"}},
        )

    if run_layercam:
        res = evaluator.evaluate_method(
            cam_method_name="LayerCAM",
            max_images=args.max_images,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            step_size=args.step_size,
            per_image_path=per_image_stem(args.output_dir, "LayerCAM"),
            road_seed=args.road_seed,
            road_imputation=args.road_imputation,
        )
        all_results.append({
            "Method": "LayerCAM",
            "Insertion_Mean": res["insertion_auc_mean"],
            "Insertion_Std": res["insertion_auc_std"],
            "Deletion_Mean": res["deletion_auc_mean"],
            "Deletion_Std": res["deletion_auc_std"],
            "ROAD_Mean": res["road_mean"],
            "ROAD_Std": res["road_std"],
            "Images_Evaluated": res["num_images"],
        })
        print(f"LayerCAM: Ins={res['insertion_auc_mean']:.4f} Del={res['deletion_auc_mean']:.4f} ROAD={res['road_mean']:.4f}")

    if run_layercam_fused:
        fuse_stages = None
        if args.fuse_stages:
            fuse_stages = [int(s) for s in args.fuse_stages.split(",")]

        stage_layers = get_layercam_stage_layers(evaluator.model, args.arch)
        print(f"\nLayerCAM-Fused: {len(stage_layers)} stages, gamma={args.gamma}, "
              f"fuse_stages={fuse_stages or 'default'}")

        image_paths, predicted_labels, class_names = evaluator.get_imagenet_images(
            max_images=args.max_images,
        )

        dataset = _ImageNetDataset(image_paths, predicted_labels, class_names, evaluator.transform)
        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset, batch_size=1, num_workers=args.num_workers,
            pin_memory=(evaluator.device.type == "cuda"), shuffle=False,
        )

        insertion_aucs, deletion_aucs, road_scores = [], [], []
        per_rows = []
        start_time = time.time()
        pbar = tqdm(loader, desc="LayerCAM-Fused", total=len(dataset))
        for i, (image_tensor, predicted_label, class_name, image_path_batch) in enumerate(pbar):
            try:
                image_tensor = image_tensor.squeeze(0)
                pred_lbl = predicted_label.item()

                saliency_map = extract_layercam_fused(
                    model=evaluator.model,
                    stage_layers=stage_layers,
                    input_tensor=image_tensor.unsqueeze(0),
                    predicted_label=pred_lbl,
                    gamma=args.gamma,
                    fuse_stages=fuse_stages,
                )

                metrics = evaluator._evaluate_saliency_map(
                    image_tensor, saliency_map, pred_lbl,
                    step_size=args.step_size, verbose=False, batch_size=args.batch_size,
                    road_seed=args.road_seed, road_imputation=args.road_imputation,
                )
                insertion_aucs.append(metrics["insertion_auc"])
                deletion_aucs.append(metrics["deletion_auc"])
                road_scores.append(metrics.get("road", np.mean([v for k, v in metrics.items() if k.startswith("road_")])))
                per_rows.append(
                    evaluator._row_from_metrics(
                        image_id=str(image_path_batch[0]),
                        method="LayerCAM-Fused",
                        predicted_label=pred_lbl,
                        metrics=metrics,
                    )
                )

                pbar.set_postfix({
                    "Ins": f"{np.mean(insertion_aucs):.3f}",
                    "Del": f"{np.mean(deletion_aucs):.3f}",
                    "ROAD": f"{np.mean(road_scores):.3f}",
                })
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue

        evaluator._write_per_image_log(per_rows, per_image_stem(args.output_dir, "LayerCAM-Fused"))
        res_fused = {
            "insertion_auc_mean": np.mean(insertion_aucs),
            "insertion_auc_std": np.std(insertion_aucs),
            "deletion_auc_mean": np.mean(deletion_aucs),
            "deletion_auc_std": np.std(deletion_aucs),
            "road_mean": np.mean(road_scores),
            "road_std": np.std(road_scores),
            "num_images": len(insertion_aucs),
        }
        all_results.append({
            "Method": "LayerCAM-Fused",
            "Insertion_Mean": res_fused["insertion_auc_mean"],
            "Insertion_Std": res_fused["insertion_auc_std"],
            "Deletion_Mean": res_fused["deletion_auc_mean"],
            "Deletion_Std": res_fused["deletion_auc_std"],
            "ROAD_Mean": res_fused["road_mean"],
            "ROAD_Std": res_fused["road_std"],
            "Images_Evaluated": res_fused["num_images"],
        })
        print(f"LayerCAM-Fused: Ins={res_fused['insertion_auc_mean']:.4f} "
              f"Del={res_fused['deletion_auc_mean']:.4f} ROAD={res_fused['road_mean']:.4f}")

    # ------------------------------------------------------------------
    # HR-CAM (multi-layer, trainable head)
    # ------------------------------------------------------------------
    if "hrcam" in methods:
        evaluator = IBSProperAUCEvaluator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            arch=args.arch,
            split=args.split,
            fold=args.fold,
            device_preference=args.device,
            layer_mode="last",
            enhanced_cam_method="GradCAMEnhanced",
            extractor_cls=EnhancedExtractorV2,
            extractor_kwargs={**extractor_kwargs_base, "aggregation_config": {"type": "standard"}},
        )
        stage_layers = get_layercam_stage_layers(evaluator.model, args.arch)

        from XAI_Enhancer_module.ibs.data import IBSDataset, IBS_NUM_CLASSES, get_train_transforms, get_val_transforms
        from torch.utils.data import DataLoader

        if args.hrcam_checkpoint:
            hrcam_head = load_hrcam_head(
                backbone=evaluator.model,
                stage_layers=stage_layers,
                num_classes=IBS_NUM_CLASSES,
                checkpoint_path=args.hrcam_checkpoint,
                device=evaluator.device,
            )
        else:
            print("\nHR-CAM: no checkpoint provided -- auto-training on train split ...")
            train_ds = IBSDataset(str(args.data_root), split="train", transform=get_train_transforms(), fold=args.fold)
            val_ds = IBSDataset(str(args.data_root), split="val", transform=get_val_transforms(), fold=args.fold)
            hr_train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(evaluator.device.type == "cuda"),
            )
            hr_val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(evaluator.device.type == "cuda"),
            )
            hr_save_path = os.path.join(args.output_dir, f"hrcam_head_{args.arch}.pth")
            hrcam_head = train_hrcam_head(
                backbone=evaluator.model,
                stage_layers=stage_layers,
                train_loader=hr_train_loader,
                num_classes=IBS_NUM_CLASSES,
                val_loader=hr_val_loader,
                epochs=args.hrcam_epochs,
                lr=args.hrcam_lr,
                device=evaluator.device,
                save_path=hr_save_path,
            )

        print(f"\nHR-CAM: {len(stage_layers)} stage layers, "
              f"total_channels={hrcam_head.total_channels}")

        image_paths, predicted_labels, class_names = evaluator.get_imagenet_images(
            max_images=args.max_images,
        )

        dataset = _ImageNetDataset(image_paths, predicted_labels, class_names, evaluator.transform)
        loader = DataLoader(
            dataset, batch_size=1, num_workers=args.num_workers,
            pin_memory=(evaluator.device.type == "cuda"), shuffle=False,
        )

        insertion_aucs, deletion_aucs, road_scores = [], [], []
        per_rows = []
        pbar = tqdm(loader, desc="HR-CAM", total=len(dataset))
        for i, (image_tensor, predicted_label, class_name, image_path_batch) in enumerate(pbar):
            try:
                image_tensor = image_tensor.squeeze(0)
                pred_lbl = predicted_label.item()

                saliency_map = extract_hrcam(
                    hrcam_head=hrcam_head,
                    input_tensor=image_tensor.unsqueeze(0),
                    predicted_label=pred_lbl,
                )

                metrics = evaluator._evaluate_saliency_map(
                    image_tensor, saliency_map, pred_lbl,
                    step_size=args.step_size, verbose=False, batch_size=args.batch_size,
                    road_seed=args.road_seed, road_imputation=args.road_imputation,
                )
                insertion_aucs.append(metrics["insertion_auc"])
                deletion_aucs.append(metrics["deletion_auc"])
                road_scores.append(metrics.get("road", np.mean([v for k, v in metrics.items() if k.startswith("road_")])))
                per_rows.append(
                    evaluator._row_from_metrics(
                        image_id=str(image_path_batch[0]),
                        method="HR-CAM",
                        predicted_label=pred_lbl,
                        metrics=metrics,
                    )
                )

                pbar.set_postfix({
                    "Ins": f"{np.mean(insertion_aucs):.3f}",
                    "Del": f"{np.mean(deletion_aucs):.3f}",
                    "ROAD": f"{np.mean(road_scores):.3f}",
                })
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue

        evaluator._write_per_image_log(per_rows, per_image_stem(args.output_dir, "HR-CAM"))
        res_hrcam = {
            "insertion_auc_mean": np.mean(insertion_aucs),
            "insertion_auc_std": np.std(insertion_aucs),
            "deletion_auc_mean": np.mean(deletion_aucs),
            "deletion_auc_std": np.std(deletion_aucs),
            "road_mean": np.mean(road_scores),
            "road_std": np.std(road_scores),
            "num_images": len(insertion_aucs),
        }
        all_results.append({
            "Method": "HR-CAM",
            "Insertion_Mean": res_hrcam["insertion_auc_mean"],
            "Insertion_Std": res_hrcam["insertion_auc_std"],
            "Deletion_Mean": res_hrcam["deletion_auc_mean"],
            "Deletion_Std": res_hrcam["deletion_auc_std"],
            "ROAD_Mean": res_hrcam["road_mean"],
            "ROAD_Std": res_hrcam["road_std"],
            "Images_Evaluated": res_hrcam["num_images"],
        })
        print(f"HR-CAM: Ins={res_hrcam['insertion_auc_mean']:.4f} "
              f"Del={res_hrcam['deletion_auc_mean']:.4f} ROAD={res_hrcam['road_mean']:.4f}")

        hrcam_head.remove_hooks()

    df = pd.DataFrame(all_results)
    out_path = os.path.join(args.output_dir, "comparison_report.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
