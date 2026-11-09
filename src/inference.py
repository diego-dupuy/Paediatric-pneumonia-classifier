"""Inference on a single X-ray: prediction, Grad-CAM heat map and image quality checks.

This is the code used by the demo app (app.py). It reuses the same preprocessing
and model definition as training, so the model sees images exactly as it learnt them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import CLASS_NAMES, IMAGE_SIZE
from .dataset import eval_transform
from .gradcam import central_fraction
from .model import build_model


@dataclass
class Prediction:
    """Everything the app and the report need about one X-ray."""

    label: str  # "Normal" or "Pneumonia"
    probability_pneumonia: float  # model output after softmax, 0-1
    threshold: float  # decision threshold used
    heatmap: np.ndarray  # Grad-CAM overlay (RGB, 224 x 224) for display
    central_attention: float  # share of the heat map in the central (lung) region
    quality_warnings: list = field(default_factory=list)  # possible problems with the input


class PneumoniaClassifier:
    """Loads the trained weights once and analyses images on demand."""

    def __init__(self, weights_path, threshold=0.5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        state = torch.load(Path(weights_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        # Grad-CAM needs gradients through the whole network (nothing is trained here)
        for param in self.model.parameters():
            param.requires_grad = True
        self.cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
        self.threshold = threshold
        self.transform = eval_transform()  # same preprocessing as validation/test

    def predict(self, image: Image.Image) -> Prediction:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        with torch.no_grad():
            scores = self.model(tensor)
            # softmax turns the two scores into probabilities that add up to 1
            prob = torch.softmax(scores, dim=1)[0, 1].item()
        label = CLASS_NAMES[1] if prob >= self.threshold else CLASS_NAMES[0]

        # Heat map of the class the model chose (0 = Normal, 1 = Pneumonia), drawn over the
        # resized X-ray: it shows which regions supported that decision
        predicted_class = CLASS_NAMES.index(label)
        heat = self.cam(input_tensor=tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]
        base = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255
        overlay = show_cam_on_image(base, heat, use_rgb=True)

        return Prediction(label, prob, self.threshold, overlay, central_fraction(heat),
                          quality_checks(image))


def quality_checks(image: Image.Image):
    """Simple checks of the input image. They do not prove it is a chest X-ray;
    they only flag inputs that are clearly unlike the training data."""
    warnings = []
    width, height = image.size
    if min(width, height) < 256:
        warnings.append(f"Low resolution ({width} x {height} px); training images were larger.")
    if not 0.5 <= width / height <= 2.0:  # the dataset includes images up to ~1.9 : 1
        warnings.append(f"Unusual aspect ratio ({width / height:.2f}); a chest X-ray is roughly square.")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    colour = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    if colour > 10:
        warnings.append("The image has colour; X-rays are greyscale. It may not be a radiograph.")
    if rgb.mean(axis=2).std() < 20:
        warnings.append("Very low contrast; the image may be over- or under-exposed.")
    return warnings
