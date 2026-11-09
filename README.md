<div align="center">

# Paediatric Pneumonia Classification on Chest X-rays

**ResNet18 (transfer learning) · PyTorch · Grad-CAM**

</div>

**Diego Dupuy Álamo** · Biomedical Engineering, Universitat Politècnica de València

Binary classifier (**Normal / Pneumonia**) for chest X-rays of children aged 1 to 5. Besides training the model,
the goal is to **understand when and why it fails**, by reviewing its errors with Grad-CAM.

It is a self-directed learning project, with special care in evaluating the model rigorously
(see [Design decisions](#design-decisions)).

**Final model on the test set:** sensitivity 95.9 %, specificity 76.9 %, AUC 0.953
(see [Results](#results)).

It includes a [demo app](#demo-app) that analyses an X-ray (PNG, JPG or DICOM), shows the Grad-CAM heat map
and generates a PDF report:

![Demo app](docs/figures/demo_app.gif)

## Data
*Chest X-Ray Images (Pneumonia)* (Kermany et al., 2018), published on Kaggle.

| Set | Normal | Pneumonia |
|---|---|---|
| Training | 1,341 | 3,875 |
| Test | 234 | 390 |

There are almost **3 pneumonia X-rays for every normal one**. This imbalance pushes the model towards answering "pneumonia".

## Method
1. **Preprocessing:** resize to 224 × 224 and normalise with the ImageNet mean and standard deviation.
2. **Patient-level validation:** the official `val` folder only has 16 images, so I hold out 15 % of the training set
   as validation, **grouped by patient** so that the same child never appears in both sets.
3. **Model:** ResNet18 pre-trained on ImageNet, with the convolutional layers frozen and a new final layer with 2 outputs.
4. **Training:** Adam (lr = 0.001), cross-entropy and *early stopping* on the validation loss (patience 4).
5. **Three experiments**, changing one thing at a time:
   * `baseline`: no class weights and no data augmentation (reference).
   * `weighted`: adds **class weights** to the loss function to compensate for the imbalance.
   * `weighted_aug`: adds **data augmentation**: random crop, ±10° rotation and brightness/contrast changes,
     without horizontal flips, because they would move the heart to the other side. In every epoch the model sees a
     different version of each X-ray, which forces it to focus on the anatomy rather than on details of specific images.

     ![Data augmentation examples](docs/figures/augmentation_examples.png)
6. **Evaluation:** sensitivity, specificity, precision, accuracy, F1 and AUC. The final model is **chosen on the
   validation set**, and the test set (624 X-rays) is only reported, so that the results are not inflated.
7. **Grad-CAM** of **all** false negatives, plus a simple measure of how much of the attention falls in the central
   region of the image rather than near the borders.

## Results
Trained on Google Colab (T4 GPU). All figures are on the **test set** (624 X-rays) with a 0.5 threshold,
except for the validation table.

| Experiment | Sensitivity | Specificity | Precision | Accuracy | F1 | AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 98.5 % | 53.0 % | 77.7 % | 81.4 % | 86.9 % | 0.936 | 110 | 6 |
| `weighted` | 98.2 % | 52.1 % | 77.4 % | 80.9 % | 86.6 % | 0.939 | 112 | 7 |
| **`weighted_aug`** | 95.9 % | **76.9 %** | **87.4 %** | **88.8 %** | **91.4 %** | **0.953** | **54** | 16 |

**Validation** (≈ 780 images split by patient), which is what the model is selected on:

| Experiment | Sensitivity | Specificity | AUC | Balanced accuracy |
|---|---|---|---|---|
| `baseline` | 98.1 % | 92.1 % | 0.995 | 95.1 % |
| `weighted` | 97.8 % | 92.1 % | 0.995 | 95.0 % |
| `weighted_aug` | 95.6 % | 95.8 % | 0.989 | **95.7 %** |

![Test confusion matrices](docs/figures/confusion_matrices.png)

### Conclusions
1. **Final model: `weighted_aug`.** It has the highest balanced accuracy on validation (mean of sensitivity and
   specificity), and the choice is made without looking at the test set.
2. **Class weights change nothing** (`baseline` and `weighted` are practically identical). In other words, the false
   positives are not caused by the class imbalance.
3. **There is a large gap between validation and test.** Specificity is 92 % on validation but drops to 53 % on test:
   the model works well on images similar to the training ones and worse on the test images, which seem to follow a
   somewhat different distribution. Without patient-level validation this would not have been detected.
4. **Data augmentation is what helps most:** it halves the false positives (110 → 54), raises specificity by
   24 points and shrinks the validation–test gap from 39 to 19 points. In exchange, there are 10 more false
   negatives (sensitivity 98.5 % → 95.9 %). Since the AUC also improves (0.936 → 0.953), the model does not just
   shift the threshold: it separates the two classes better.
5. With data augmentation the validation loss was still decreasing at epoch 15, so training for more epochs could
   improve the result slightly.

![Training curves](docs/figures/training_curves.png)

### Grad-CAM: where does the model look when it fails?
For each pneumonia X-ray, I compute the fraction of the Grad-CAM map that falls in the central region of the image
(removing a 15 % margin on each side). If the map were uniform, that fraction would be **49 %**.

| | `baseline` | `weighted_aug` |
|---|---|---|
| False negatives | 6 | 16 |
| Central attention (all pneumonia cases) | 50.7 % | 56.1 % |
| Central attention (false negatives only) | **35.2 %** | **45.5 %** |

* **In both models, false negatives look at the centre less than average** (and below the 49 % of a uniform map).
  In many of these cases the attention is on the abdomen, the neck or the corners, outside the lungs.
* **Data augmentation increases central attention** (50.7 % → 56.1 %): with random crops, the information near the
  borders is no longer reliable, so the model relies more on the lung region.
* It is a simple measure: the central rectangle does not match the lungs exactly and the Grad-CAM map has a low
  resolution (7 × 7), so it shows trends, not individual cases.

![Grad-CAM of the false negatives (weighted_aug)](docs/figures/gradcam_fn_weighted_aug.png)

<details>
<summary>Grad-CAM of the <code>baseline</code> false negatives</summary>

![Grad-CAM of the false negatives (baseline)](docs/figures/gradcam_fn_baseline.png)
</details>

### Limitations and possible improvements
* A single dataset from a single centre; there is no external validation.
* The threshold is fixed at 0.5. It could be tuned on validation to recover sensitivity.
* Only the last layer is trained. Fine-tuning the last ResNet block (`layer4`) as well, or segmenting the lungs
  before classifying, are the natural next steps.

## Design decisions
* **Patient-level validation:** the official `val` folder only has 16 images, not enough for *early stopping*.
  So I use about 780 training images, split by patient to avoid data leakage.
* **Complete metrics:** with imbalanced classes, sensitivity alone is misleading (always answering "pneumonia"
  gives 100 %). That is why I also report specificity, precision, accuracy and F1.
* **One change per experiment:** this way the effect of class weights and of data augmentation can be measured separately.
* **Grad-CAM of all false negatives**, not of a single hand-picked case, plus a numeric measure of where the model looks.
* **Modular code** (`src/`) and a notebook only for training on Colab with a GPU. Fixed random seed for reproducibility,
  and the Kaggle key is read from Colab *secrets*, never from the code.

## Structure
```
src/
├── config.py      # paths, hyperparameters and definition of the experiments
├── dataset.py     # transforms, data augmentation, patient-level split, DataLoaders
├── model.py       # ResNet18 with transfer learning
├── train.py       # training loop with early stopping
├── evaluate.py    # test metrics, confusion matrix, comparison of experiments
├── gradcam.py     # Grad-CAM of the false negatives and central attention measure
├── inference.py   # analysis of one X-ray: prediction, Grad-CAM and quality checks
├── dicom_utils.py # DICOM reading with prior anonymisation
└── report.py      # PDF report from a fixed template
app.py             # demo web app (Gradio)
examples/          # public example X-rays for the app
notebooks/
└── train_colab.ipynb   # runs everything on Google Colab (GPU)
```

## Demo app
`app.py` is a web interface to use the trained model end to end:
1. **Input:** upload an X-ray in PNG, JPG or **DICOM**. For DICOM, the patient's identifying data (name, ID, dates,
   institution… and private tags) is removed before anything else.
2. **Analysis:** the model returns the probability of pneumonia and the **Grad-CAM map** of the predicted class
   (which regions weighed most in the decision), together with basic image checks (resolution, aspect ratio,
   colour, contrast).
3. **PDF report** generated from a fixed template: result, images, quality checks, model performance on the test set
   and limitations.

> ⚠️ Technical demo: **not a medical device** and not suitable for clinical use. Use it only with public or anonymised images.

### Run the app on your computer
You only need Python (3.10 or later); no GPU required.

```bash
git clone https://github.com/diego-dupuy/Paediatric-pneumonia-classifier.git
cd Paediatric-pneumonia-classifier
pip install -r requirements.txt
python app.py
```

Then open the address shown in the terminal (usually http://127.0.0.1:7860) in your browser.
The first time, the app downloads the trained model (45 MB) from the
[repository release](https://github.com/diego-dupuy/Paediatric-pneumonia-classifier/releases/tag/modelo).
The weights are not stored with the code because Git keeps every version of every file, so the repository would
grow with each retraining.

The `examples/` folder contains 4 X-rays from the public test set (Kermany et al., 2018; CC BY 4.0 licence) and a
DICOM file with **made-up** patient data, to try the anonymisation.

## How to retrain the model
Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab, select a T4 GPU, add your Kaggle key
as a *secret* (`KAGGLE_API_TOKEN`) and run all the cells. Locally (with a GPU):

```bash
pip install -r requirements.txt
python -m src.train --experiment weighted_aug
python -m src.evaluate --experiment weighted_aug
python -m src.gradcam --experiment weighted_aug
```

## References
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., … Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5), 1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

Mooney, P. (2018). *Chest X-Ray Images (Pneumonia)* [Data set]. Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74
