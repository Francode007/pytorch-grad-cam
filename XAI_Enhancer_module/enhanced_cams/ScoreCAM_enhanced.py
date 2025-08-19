from torch.nn.modules import Module
from pytorch_grad_cam.base_cam import BaseCAM
import numpy as np
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import tqdm
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from typing import List, Callable, Optional

class ScoreCAMEnhanced(BaseCAM):
    def __init__(self, 
                model, 
                target_layers,
                reshape_transform=None):
        super(ScoreCAMEnhanced, self).__init__(model, target_layers, reshape_transform=reshape_transform, uses_gradients=False)
        self.target_layers = target_layers
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
    
    def normalize_and_mask_activations(self, 
                                       weighted_activations: np.ndarray, 
                                       activations: np.ndarray) -> np.ndarray:
        """
        Channel-wise normalize weighted_activations to [0, 1] and multiply with activations to get masked_activations.
        Handles both 4D (N, C, H, W) and 5D (N, C, D, H, W) arrays.
        """
        # Compute min and max per channel (axis=(2,3) for 2D, axis=(2,3,4) for 3D)
        if weighted_activations.ndim == 4:
            # (N, C, H, W)
            min_val = weighted_activations.min(axis=(2, 3), keepdims=True)
            max_val = weighted_activations.max(axis=(2, 3), keepdims=True)
        elif weighted_activations.ndim == 5:
            # (N, C, D, H, W)
            min_val = weighted_activations.min(axis=(2, 3, 4), keepdims=True)
            max_val = weighted_activations.max(axis=(2, 3, 4), keepdims=True)
        else:
            raise ValueError("weighted_activations must be 4D or 5D array.")
        denom = max_val - min_val
        denom[denom < 1e-8] = 1.0  # avoid division by zero
        norm_weighted = (weighted_activations - min_val) / denom
        masked_activations = norm_weighted * activations
        return masked_activations

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to(self.device)

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            input_tensors = input_tensor[:, None,
                                         :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights

    def get_cam_image(self, 
                      input_tensor: torch.Tensor, 
                      target_layer: Module, 
                      targets: List[Module], 
                      activations: torch.Tensor, 
                      grads: torch.Tensor, 
                      eigen_smooth: bool = False) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()
        
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        
        ### normalize the weighted_activations and create masked_activations
        masked_activations = self.normalize_and_mask_activations(weighted_activations, activations)
        return cam, masked_activations
    
    def compute_cam_per_layer(self, 
                              input_tensor: torch.Tensor, 
                              targets: List[Module], 
                              eigen_smooth: bool) -> np.ndarray:
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        intermediate_act_per_target_layer = []
        
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam, masked_activations = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
            
            # Handle masked_activations dimensions correctly
            if masked_activations.ndim == 4:
                # For 4D: (batch, channels, H, W) -> keep as is for the intermediate output
                intermediate_act_per_target_layer.append(masked_activations)
            elif masked_activations.ndim == 5:
                # For 5D: (batch, channels, D, H, W) -> keep as is for the intermediate output  
                intermediate_act_per_target_layer.append(masked_activations)
            else:
                # Fallback: just append as is
                intermediate_act_per_target_layer.append(masked_activations)

        return cam_per_target_layer, intermediate_act_per_target_layer
    
    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph = True, create_graph = True)
            if 'hpu' in str(self.device):
                self.__htcore.mark_step()

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer, mod_act_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return cam_per_layer, mod_act_per_layer
    
    def forward_augmentation_smoothing(self, input_tensor: torch.Tensor, targets: List[Module], eigen_smooth: bool = False) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam_per_layer, mod_act = self.forward(augmented_tensor, targets, eigen_smooth)
            cam = self.aggregate_multi_layers(cam_per_layer)
            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam, mod_act
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth)
