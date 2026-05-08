# Brain MRI Segmentation — U-Net for Tumor Detection

A medical image segmentation project using a custom **U-Net** architecture to detect and segment brain tumors from MRI scans. Trained on the LGG MRI Segmentation dataset with Dice Loss and IoU metrics, achieving 90%+ Dice coefficient on the test set.

---

## Highlights

- **Custom U-Net Architecture** — A full encoder-decoder U-Net built from scratch with skip connections, BatchNormalization, and progressive feature extraction (64 → 128 → 256 → 512 → 1024 filters), designed specifically for biomedical image segmentation.
- **Dice Loss Optimization** — Uses Dice Loss as the primary training objective instead of standard cross-entropy, directly optimizing for the overlap between predicted and ground-truth tumor masks — the metric that matters most in medical segmentation.
- **90%+ Dice Coefficient** — Achieves a Dice coefficient exceeding 0.90 on the test set, demonstrating strong tumor boundary detection even on challenging MRI slices with irregular tumor shapes and low contrast.
- **Bounding Box Visualization** — Post-processing pipeline that converts predicted segmentation masks into bounding boxes drawn on the original MRI, providing clinicians with both the pixel-level segmentation and a quick-glance tumor localization.
- **Comprehensive Evaluation** — Training curves for Accuracy, IoU, and Dice Coefficient; side-by-side comparison of original MRI, ground truth mask, and predicted mask; and per-sample visual predictions on 20 random test images.

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │   Brain MRI Image (256x256) │
                    └────────────┬────────────┘
                                 │
                                 ▼
            ┌─────────────────────────────────────┐
            │          U-Net ENCODER               │
            │  Conv2D(64) → BN → ReLU → MaxPool   │
            │  Conv2D(128) → BN → ReLU → MaxPool  │
            │  Conv2D(256) → BN → ReLU → MaxPool  │
            │  Conv2D(512) → BN → ReLU → MaxPool  │
            └────────────┬────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────────┐
            │          BOTTLENECK                   │
            │  Conv2D(1024) → BN → ReLU → Dropout  │
            └────────────┬────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────────┐
            │          U-Net DECODER               │
            │  UpSampling + Concat (skip conn)     │
            │  Conv2D(512) → BN → ReLU             │
            │  UpSampling + Concat (skip conn)     │
            │  Conv2D(256) → BN → ReLU             │
            │  UpSampling + Concat (skip conn)     │
            │  Conv2D(128) → BN → ReLU             │
            │  UpSampling + Concat (skip conn)     │
            │  Conv2D(64) → BN → ReLU              │
            └────────────┬────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────────┐
            │  Conv2D(1) → Sigmoid → Tumor Mask    │
            └────────────┬────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────────┐
            │     Post-Processing                   │
            │  • Binary threshold (0.5)             │
            │  • Contour detection                  │
            │  • Bounding box overlay on MRI        │
            └──────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | TensorFlow / Keras |
| **Architecture** | U-Net (custom implementation) |
| **Optimizer** | Adamax (lr=0.001) |
| **Loss Function** | Dice Loss |
| **Metrics** | Dice Coefficient, IoU Coefficient, Accuracy |
| **Data Pipeline** | Keras ImageDataGenerator |
| **Image Processing** | OpenCV, scikit-image |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | LGG MRI Segmentation (Kaggle) |
| **Environment** | Kaggle Notebooks / Google Colab |

---

## Project Structure

```
brain-mri-segmentation-cv3/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── notebooks/
│   ├── final.ipynb              # Complete training & evaluation pipeline
│   ├── finally_last.ipynb       # Full pipeline with prediction visualizations
│   ├── done_dice_90.ipynb       # Best model — 90%+ Dice score
│   └── done_dice_90_keras3.ipynb # Keras 3 compatible version
```

---

## How It Works

### 1. Data Loading & Splitting

The dataset is loaded from the LGG MRI Segmentation archive, where each patient folder contains MRI slices and their corresponding tumor masks (suffixed with `_mask`). A DataFrame is built by pairing each image with its mask path, then split into **80% training**, **10% validation**, and **10% test** sets using stratified splitting to maintain consistent tumor prevalence across splits.

### 2. Data Augmentation

Training images and masks are augmented simultaneously using Keras `ImageDataGenerator` with the same random seed to maintain spatial alignment. Augmentations include rotation (±0.2 radians), width and height shifts (5%), shear (5%), zoom (5%), and horizontal flipping. All images are resized to 256x256 and normalized to [0, 1]. A custom generator yields (image, mask) pairs for each batch, ensuring the augmentation transforms are applied identically to both.

### 3. U-Net Architecture

The U-Net is built from scratch following the original Ronneberger et al. design. The **encoder** consists of 4 downsampling blocks, each with two Conv2D layers (3x3 kernel, same padding), BatchNormalization, ReLU activation, and MaxPooling2D (2x2), progressively increasing filters from 64 to 512. The **bottleneck** uses two Conv2D layers with 1024 filters and Dropout for regularization. The **decoder** mirrors the encoder with 4 upsampling blocks, each using UpSampling2D (2x2), concatenation with the corresponding encoder skip connection, and two Conv2D layers with decreasing filters. The final layer is a Conv2D with 1 filter and Sigmoid activation to produce a probability map for the tumor region.

### 4. Training Strategy

The model is compiled with the **Adamax optimizer** (learning rate 0.001) and **Dice Loss** as the objective function. Dice Loss is the negative of the Dice Coefficient, which directly measures the overlap between predicted and ground-truth masks. Three metrics are tracked: Accuracy, IoU Coefficient, and Dice Coefficient. Training runs for up to 150 epochs with a batch size of 40, using `ModelCheckpoint` to save the best model (lowest validation loss). Early convergence is typically observed around 50–80 epochs.

### 5. Evaluation & Visualization

After training, the model is evaluated on the train, validation, and test sets with all three metrics reported. The evaluation pipeline includes: training history plots (Accuracy, IoU, Dice, Loss curves with best epoch highlighted), side-by-side comparison of 10 test samples showing original MRI, ground truth mask, and predicted mask with bounding box overlay, and 20 random test predictions for qualitative assessment.

### 6. Post-Processing: Bounding Box Extraction

Predicted masks are binarized with a 0.5 threshold, and OpenCV contour detection is applied to find tumor regions. Bounding rectangles are computed for each contour and drawn on the original MRI image, providing a quick visual summary of tumor location and extent. This dual representation — pixel-level segmentation plus bounding box — makes the output useful for both clinical review and integration with downstream analysis pipelines.

---

## Key Metrics

| Metric | Description |
|--------|------------|
| **Dice Coefficient** | Measures overlap between predicted and ground truth masks (2 × intersection / union). Primary evaluation metric. Achieved **90%+** |
| **IoU Coefficient** | Intersection over Union — area of overlap divided by area of union. Standard segmentation benchmark |
| **Accuracy** | Pixel-wise classification accuracy across the entire image |
| **Dice Loss** | Negative Dice Coefficient, used as the training objective to directly optimize segmentation quality |

---

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.12+
- Kaggle account (to download the dataset)

### Installation

```bash
# Clone the repository
git clone https://github.com/aliisnetalive/brain-mri-segmentation-cv3.git
cd brain-mri-segmentation-cv3

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the **LGG MRI Segmentation** dataset from Kaggle:

```bash
# Using Kaggle CLI
pip install kaggle
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d kaggle_3m
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) and extract to the project root.

### Run the Notebook

Open any notebook in `notebooks/` in **Kaggle**, **Google Colab**, or **Jupyter** and update the `data_dir` path:

```python
data_dir = '/kaggle/input/lgg-mri-segmentation/kaggle_3m'
# or for local:
data_dir = './kaggle_3m'
```

Then run all cells to train the model and view results.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 256 x 256 x 3 |
| Batch Size | 40 |
| Epochs | 150 (with ModelCheckpoint) |
| Optimizer | Adamax (lr=0.001) |
| Loss Function | Dice Loss |
| Metrics | Accuracy, IoU Coefficient, Dice Coefficient |
| Data Split | 80% Train / 10% Val / 10% Test |

---

## U-Net Architecture Details

```
Layer (type)                    Output Shape         Param #
================================================================
input_1 (InputLayer)            [(None, 256, 256, 3)]    0
________________________________________________________________
conv2d (Conv2D)                 (None, 256, 256, 64)   1792
batch_normalization (BN)        (None, 256, 256, 64)   256
activation (ReLU)               (None, 256, 256, 64)   0
conv2d_1 (Conv2D)               (None, 256, 256, 64)   36928
batch_normalization_1 (BN)      (None, 256, 256, 64)   256
activation_1 (ReLU)             (None, 256, 256, 64)   0
max_pooling2d (MaxPool)         (None, 128, 128, 64)   0
________________________________________________________________
... (encoder continues: 128 → 256 → 512)
________________________________________________________________
conv2d_8 (Conv2D)               (None, 16, 16, 1024)   9438208  (bottleneck)
dropout (Dropout)               (None, 16, 16, 1024)   0
________________________________________________________________
... (decoder with skip connections: 512 → 256 → 128 → 64)
________________________________________________________________
conv2d_18 (Conv2D)              (None, 256, 256, 1)    577
activation_sigmoid (Sigmoid)    (None, 256, 256, 1)    0
================================================================
```

---

## Dataset Details

- **Source**: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) by Mateusz Buda
- **Description**: Brain MRI images with manual FLAIR abnormality segmentation masks from The Cancer Imaging Archive (TCIA)
- **Patients**: 110 patients with lower grade glioma
- **Slices**: ~3,900 MRI slices with paired tumor masks
- **Format**: PNG images (256x256) with `_mask` suffix convention
- **Labels**: Binary segmentation (tumor vs. no tumor)

---

## Dependencies

```
tensorflow>=2.12.0
keras>=3.0.0
numpy
pandas
opencv-python
matplotlib
seaborn
scikit-learn
scikit-image
tqdm
```

---

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) — Kaggle
- [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)

---

## License

This project is open source and available for educational and personal use. The LGG MRI Segmentation dataset is available through Kaggle and The Cancer Imaging Archive.
