from torch.nn.modules import Module
from pytorch_grad_cam.base_cam import BaseCAM
import numpy as np
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import tqdm
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.find_layers import replace_layer_recursive
from pytorch_grad_cam.ablation_layer import AblationLayer
from typing import List, Callable, Optional

class AblationCAMEnhanced(BaseCAM):
    def __init__(self, 
                model, 
                target_layers,
                reshape_transform=None,
                ablation_layer=AblationLayer(),
                batch_size=32,
                ratio_channels_to_ablate=1.0):
        super(AblationCAMEnhanced, self).__init__(model, target_layers, reshape_transform=reshape_transform, uses_gradients=False)
        self.target_layers = target_layers
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.batch_size = batch_size
        self.ablation_layer = ablation_layer
        self.ratio_channels_to_ablate = ratio_channels_to_ablate
        self.activations = None
    
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

    def save_activation(self, module, input, output) -> None:
        """ Helper function to save the raw activations from the target layer """
        self.activations = output

    def assemble_ablation_scores(self,
                                 new_scores: list,
                                 original_score: float,
                                 ablated_channels: np.ndarray,
                                 number_of_channels: int) -> np.ndarray:
        """ Take the value from the channels that were ablated,
            and just set the original score for the channels that were skipped """

        index = 0
        result = []
        sorted_indices = np.argsort(ablated_channels)
        ablated_channels = ablated_channels[sorted_indices]
        new_scores = np.float32(new_scores)[sorted_indices]

        for i in range(number_of_channels):
            if index < len(ablated_channels) and ablated_channels[index] == i:
                weight = new_scores[index]
                index = index + 1
            else:
                weight = original_score
            result.append(weight)

        return result

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        # Do a forward pass, compute the target scores, and cache the activations
        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            handle.remove()
            original_scores = np.float32(
                [target(output).cpu().item() for target, output in zip(targets, outputs)])

        # Replace the layer with the ablation layer.
        # When we finish, we will replace it back, so the 
        # original model is unchanged.
        ablation_layer = self.ablation_layer
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        number_of_channels = activations.shape[1]
        weights = []
        # This is a "gradient free" method, so we don't need gradients here.
        with torch.no_grad():
            # Loop over each of the batch images and ablate activations for it.
            for batch_index, (target, tensor) in enumerate(
                    zip(targets, input_tensor)):
                new_scores = []
                batch_tensor = tensor.repeat(self.batch_size, 1, 1, 1)

                # Check which channels should be ablated. Normally this will be all channels,
                # But we can also try to speed this up by using a low
                # ratio_channels_to_ablate.
                channels_to_ablate = ablation_layer.activations_to_be_ablated(
                    activations[batch_index, :], self.ratio_channels_to_ablate)
                number_channels_to_ablate = len(channels_to_ablate)

                for i in tqdm.tqdm(
                    range(
                        0,
                        number_channels_to_ablate,
                        self.batch_size)):
                    if i + self.batch_size > number_channels_to_ablate:
                        batch_tensor = batch_tensor[:(
                            number_channels_to_ablate - i)]

                    # Change the state of the ablation layer so it ablates the next channels.
                    # TBD: Move this into the ablation layer forward pass.
                    ablation_layer.set_next_batch(
                        input_batch_index=batch_index,
                        activations=self.activations,
                        num_channels_to_ablate=batch_tensor.size(0))
                    score = [target(o).cpu().item()
                             for o in self.model(batch_tensor)]
                    new_scores.extend(score)
                    ablation_layer.indices = ablation_layer.indices[batch_tensor.size(
                        0):]

                new_scores = self.assemble_ablation_scores(
                    new_scores,
                    original_scores[batch_index],
                    channels_to_ablate,
                    number_of_channels)
                weights.extend(new_scores)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        # Replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, target_layer)
        # Returning the weights from new_scores
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
