"""TMENet: joint regulatory-network and spatial-coupling inference."""
from .model import FitResult, fit_generalized
from .preprocessing import prepare_inputs, spatial_neighborhood
from .lr import attribute_lr

__all__ = ["FitResult", "fit_generalized", "prepare_inputs", "spatial_neighborhood", "attribute_lr"]
__version__ = "0.1.0"
