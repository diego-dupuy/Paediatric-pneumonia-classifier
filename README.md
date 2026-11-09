<div align="center">

# Clasificación de neumonía pediátrica en radiografías de tórax

**ResNet18 (transfer learning) · PyTorch · Grad-CAM**

</div>

Clasificador binario (**Sano / Neumonía**) de radiografías de tórax de niños de 1 a 5 años. Además de entrenar el modelo,
el objetivo es **entender cuándo y por qué falla**, revisando sus errores con Grad-CAM.

Es un proyecto de aprendizaje autónomo, con especial cuidado en evaluar el modelo de forma rigurosa
(ver [Decisiones de diseño](#decisiones-de-diseño)).

## Datos
*Chest X-Ray Images (Pneumonia)* (Kermany et al., 2018), publicado en Kaggle.

| Conjunto | Sanas | Neumonía |
|---|---|---|
| Entrenamiento | 1 341 | 3 875 |
| Test | 234 | 390 |

Hay casi **3 radiografías con neumonía por cada sana**. Este desequilibrio hace que el modelo tienda a decir "neumonía".

## Método
1. **Preprocesado:** redimensionado a 224 × 224 y normalización con la media y desviación de ImageNet.
2. **Validación por paciente:** la carpeta `val` oficial solo tiene 16 imágenes, así que separo un 15 % de
   entrenamiento como validación, **agrupando por paciente** para que un mismo niño no aparezca en los dos conjuntos.
3. **Modelo:** ResNet18 preentrenada en ImageNet, con las capas convolucionales congeladas y una última capa nueva de 2 salidas.
4. **Entrenamiento:** Adam (lr = 0.001), CrossEntropy y *early stopping* sobre la pérdida de validación (paciencia 4).
5. **Tres experimentos**, cambiando una sola cosa cada vez:
   * `baseline`: sin pesos por clase ni data augmentation (referencia).
   * `weighted`: añade **pesos por clase** en la función de pérdida para compensar el desequilibrio.
   * `weighted_aug`: añade **data augmentation**: recorte aleatorio, rotación de ±10° y cambios de brillo y contraste,
     sin volteo horizontal, porque cambiaría el lado del corazón.
6. **Evaluación:** sensibilidad, especificidad, precisión, exactitud, F1 y AUC. El modelo final se **elige con
   la validación** y el test (624 radiografías) solo se reporta, para no inflar los resultados.
7. **Grad-CAM** de **todos** los falsos negativos, más una medida sencilla de cuánta atención cae en la zona central
   de la imagen frente a los bordes.

## Resultados
*Pendiente: se añaden tras el entrenamiento en Colab.*

## Decisiones de diseño
* **Validación por paciente:** la carpeta `val` oficial solo tiene 16 imágenes, insuficientes para el *early stopping*.
  Por eso uso unas 780 imágenes de entrenamiento, separadas por paciente para evitar fuga de datos.
* **Métricas completas:** con clases desequilibradas, la sensibilidad sola engaña (decir siempre "neumonía" da un 100 %).
  Por eso reporto también especificidad, precisión, exactitud y F1.
* **Un cambio por experimento:** así se puede medir el efecto de los pesos por clase y el del data augmentation por separado.
* **Grad-CAM de todos los falsos negativos**, no de un caso aislado, más una medida numérica de dónde mira el modelo.
* **Código modular** (`src/`) y un notebook solo para entrenar en Colab con GPU. Semilla fija para reproducibilidad,
  y la clave de Kaggle se lee de los *secrets* de Colab, nunca del código.

## Estructura
```
src/
├── config.py      # rutas, hiperparámetros y definición de los experimentos
├── dataset.py     # transformaciones, data augmentation, separación por paciente, DataLoaders
├── model.py       # ResNet18 con transfer learning
├── train.py       # bucle de entrenamiento con early stopping
├── evaluate.py    # métricas en test, matriz de confusión, comparación de experimentos
├── gradcam.py     # Grad-CAM de los falsos negativos y medida de atención central
├── inference.py   # análisis de una radiografía: predicción, Grad-CAM y controles de calidad
├── dicom_utils.py # lectura de DICOM con anonimización previa
└── report.py      # informe PDF a partir de una plantilla fija
app.py             # aplicación web de demostración (Gradio)
notebooks/
└── train_colab.ipynb   # ejecuta todo en Google Colab (GPU)
```

## Aplicación de demostración
`app.py` es una interfaz web para usar el modelo entrenado de principio a fin:
1. **Entrada:** se sube una radiografía en PNG, JPG o **DICOM**. En DICOM se eliminan antes de nada los datos
   identificativos del paciente (nombre, ID, fechas, centro… y etiquetas privadas).
2. **Análisis:** el modelo devuelve la probabilidad de neumonía y el **mapa Grad-CAM**, junto con unos controles
   básicos de la imagen (resolución, proporción, color, contraste).
3. **Informe PDF** generado con una plantilla fija: resultado, imágenes, controles de calidad, rendimiento del
   modelo en test y limitaciones.

> ⚠️ Demostración técnica: **no es un producto sanitario** ni apto para uso clínico. Úsala solo con imágenes públicas o anonimizadas.

```bash
python app.py   # usa results/weighted_aug/model.pth (o la ruta indicada en MODEL_PATH)
```

## Cómo ejecutarlo
Abre [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) en Colab, elige GPU T4, añade tu clave de Kaggle
como *secret* (`KAGGLE_API_TOKEN`) y ejecuta todas las celdas. En local:

```bash
pip install -r requirements.txt
python -m src.train --experiment weighted_aug
python -m src.evaluate --experiment weighted_aug
python -m src.gradcam --experiment weighted_aug
```

## Bibliografía
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. En *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., … Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5), 1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

Mooney, P. (2018). *Chest X-Ray Images (Pneumonia)* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. En *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74
