"""Demo web app: upload a chest X-ray (PNG/JPG or DICOM) and get the model output,
the Grad-CAM heat map and a PDF report.

Run locally:   python app.py   (then open the address shown in the terminal)
Model weights: results/<experiment>/model.pth produced by src/train.py,
               or another path given in the MODEL_PATH environment variable.
               If the file does not exist, the trained model is downloaded from
               the GitHub release of this repository the first time the app runs.
"""

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.config import IMAGE_SIZE, RESULTS_DIR
from src.dicom_utils import read_dicom
from src.inference import PneumoniaClassifier
from src.report import DISCLAIMER, build_report

EXPERIMENT = os.environ.get("EXPERIMENT", "weighted_aug")  # model used by the demo
MODEL_PATH = Path(os.environ.get("MODEL_PATH", RESULTS_DIR / EXPERIMENT / "model.pth"))
METRICS_PATH = MODEL_PATH.parent / "metrics.json"  # test-set metrics, if available

# The trained weights are not stored in the code (45 MB binary file): they are attached
# to a GitHub release and downloaded once if they are not on this computer yet.
RELEASE_URL = ("https://github.com/diego-dupuy/Clasificador-Neumonia-Pediatrica"
               "/releases/download/modelo")
for path in (MODEL_PATH, METRICS_PATH):
    if not path.exists():
        print(f"Downloading {path.name} from the GitHub release...")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(f"{RELEASE_URL}/{path.name}", str(path))

classifier = PneumoniaClassifier(MODEL_PATH)
model_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


def analyse(file_path):
    """Called by the interface every time a file is uploaded."""
    if file_path is None:
        return None, None, "Upload a chest X-ray to start.", None

    dicom_fields_removed = None
    if file_path.lower().endswith(".dcm"):
        _, image, dicom_fields_removed = read_dicom(file_path)
    else:
        image = Image.open(file_path)

    prediction = classifier.predict(image)
    original = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))

    report_path = Path(tempfile.mkdtemp()) / "xray_report.pdf"
    build_report(report_path, prediction, original, Path(file_path).name,
                 model_metrics, dicom_fields_removed)

    summary = [
        f"### Model output: **{prediction.label}**",
        f"Probability of pneumonia: **{prediction.probability_pneumonia:.1%}** "
        f"(threshold {prediction.threshold:.0%})",
        f"Attention in the central (lung) region: {prediction.central_attention:.0%}",
    ]
    if dicom_fields_removed is not None:
        summary.append(f"DICOM: {dicom_fields_removed} identifying fields removed before analysis.")
    for warning in prediction.quality_warnings:
        summary.append(f"⚠️ {warning}")
    summary.append(f"\n*{DISCLAIMER}*")

    probabilities = {"Pneumonia": prediction.probability_pneumonia,
                     "Normal": 1 - prediction.probability_pneumonia}
    return probabilities, prediction.heatmap, "\n\n".join(summary), str(report_path)


with gr.Blocks(title="Paediatric pneumonia classifier") as demo:
    gr.Markdown("# Paediatric pneumonia classifier (chest X-ray)\n"
                "ResNet18 + Grad-CAM · **Technical demo, not for clinical use.** "
                "Use only public or anonymised images.")
    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Chest X-ray (PNG, JPG or DICOM .dcm)",
                             file_types=[".png", ".jpg", ".jpeg", ".dcm"], type="filepath")
            label = gr.Label(label="Model output")
            details = gr.Markdown()
        with gr.Column():
            heatmap = gr.Image(label="Grad-CAM: regions that most influenced the model's output")
            report = gr.File(label="PDF report")
    upload.change(analyse, inputs=upload, outputs=[label, heatmap, details, report])

    # Public example images from the test set of Kermany et al. (2018), CC BY 4.0.
    # The DICOM file contains made-up patient data to show the anonymisation step.
    examples_dir = Path(__file__).parent / "examples"
    examples = sorted(str(p) for p in examples_dir.glob("*") if p.suffix in {".jpeg", ".dcm"})
    if examples:
        gr.Examples(examples=examples, inputs=upload, label="Examples (click one to analyse it)")

if __name__ == "__main__":
    demo.launch()
