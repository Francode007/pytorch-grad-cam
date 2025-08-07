import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

from XAI_Enhancer_module.metrics.utils_metric import *
from pytorch_grad_cam.metrics.road import ROADCombined

n_classes = 2

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, model_name, mode, step, substrate_fn, device_preference="auto"):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            device_preference (str): device preference for computation.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.model_name = model_name
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.device_preference = device_preference
        
        # Import get_device here to avoid circular imports
        from XAI_Enhancer_module.model_utils import get_device
        self.device = get_device(device_preference)

    def HW_calc(self, model_name):
        HW = 224 * 224
        if model_name in ('b4'):
            HW = 384 * 384
        return HW

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        HW = self.HW_calc(self.model_name)
        pred = self.model(img_tensor.to(self.device))
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1).copy()
        for i in range(n_steps+1):
            pred = self.model(start.to(self.device))
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c].detach().cpu().numpy()
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                # Convert coords to tensor indices for proper tensor assignment
                coords_tensor = torch.from_numpy(coords).long()
                start_flat = start.view(1, 3, HW)
                finish_flat = finish.view(1, 3, HW)
                start_flat[0, :, coords_tensor[0]] = finish_flat[0, :, coords_tensor[0]]
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        HW = self.HW_calc(self.model_name)
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i*batch_size:(i+1)*batch_size].to(self.device)).cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions.detach().numpy(), -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1).copy()
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j*batch_size:(j+1)*batch_size].to(self.device))
                preds = preds.cpu().detach().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            # Convert coords to tensor indices for proper tensor assignment
            coords_tensor = torch.from_numpy(coords).long()
            start_flat = start.view(n_samples, 3, HW)
            finish_flat = finish.view(n_samples, 3, HW)
            for sample_idx in range(n_samples):
                start_flat[sample_idx, :, coords_tensor[sample_idx]] = finish_flat[sample_idx, :, coords_tensor[sample_idx]]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores
    
def get_metrics(model, model_name, img_size=224, device_preference="auto"):
    # substrate fn
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    insertion = CausalMetric(model, model_name, 'ins', img_size, substrate_fn=blur, device_preference=device_preference)
    deletion = CausalMetric(model, model_name, 'del', img_size, substrate_fn=torch.zeros_like, device_preference=device_preference)
    road_combined = ROADCombined(percentiles=[20, 40, 60, 80])
    return insertion, deletion, road_combined