import torch

from typing import Callable, Optional

from torch import nn
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

class GradCAM:
    _forward_handle: Optional[RemovableHandle]
    _grad_handle: Optional[RemovableHandle]
    _grads: Optional[torch.Tensor]

    def __init__(self, model: nn.Module, layer_getter: Callable[[nn.Module], nn.Module]):
        self._model = model

        self._forward_handle = None
        self._grad_handle = None
        self._grads = None
        self._layer_getter = layer_getter

    def __call__(
        self, 
        x: torch.Tensor
    ):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        self._model.eval()

        layer_of_interest = self._layer_getter(self._model)

        self._forward_handle = layer_of_interest.register_forward_hook(
            self._register_forward_hook
        )

        preds = self._model(x)
        index = preds.argmax(dim=-1)

        preds[:, index].backward()

        pooled_grads = self._grads.mean(dim=[0,2,3])
        activations = self._layer_activations.detach()

        heatmap = activations.clone()

        for i in range(heatmap.size(1)):
            heatmap[:,i,:,:] *= pooled_grads[i]

        heatmap = heatmap.mean(dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= heatmap.max()

        self._forward_handle.remove()

        return heatmap, activations, self._grads

    def _register_forward_hook(
        self, module: nn.Module, 
        input: torch.Tensor, 
        output: torch.Tensor
    ) -> None:
        self._grad_handle = output.register_hook(self._grad_capture_hook)
        self._layer_activations = output

    def _grad_capture_hook(self, grads: torch.Tensor) -> None:
        self._grads = grads
