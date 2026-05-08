# Brain MRI Segmentation — U-Net + Bounding Box | 80% Dice & IoU

A medical image segmentation project that uses a custom **U-Net** convolutional neural network to detect and segment Lower-Grade Gliomas (LGG) in brain MRI scans. The model achieves **80% Dice Coefficient and IoU** on the test set, and includes a post-processing pipeline that draws bounding boxes around detected tumors for quick clinical interpretation.

---

## Highlights

- **Custom U-Net from Scratch** — A full encoder-decoder architecture built from the ground up with skip connections, BatchNormalization, and Conv2DTranspose upsampling (64 → 128 → 256 → 512 → 1024 filters), following the classic Ronneberger et al. design optimized for biomedical image segmentation.
- **Dice Loss Optimization** — Uses Dice Loss as the primary training objective instead of standard cross-entropy, directly optimizing for the overlap between predicted and ground-truth tumor masks — the metric that matters most in medical segmentation tasks.
- **80% Dice & IoU** — Achieves a Dice coefficient and IoU of approximately 0.80 on the test set, demonstrating reliable tumor boundary detection on MRI slices with irregular tumor shapes and varying contrast levels.
- **Bounding Box Visualization** — A dedicated post-processing function converts predicted segmentation masks into bounding boxes drawn on the original MRI using OpenCV contour detection. This provides clinicians with both pixel-level segmentation and a quick-glance tumor localization in a single view.
- **All-in-One Notebook** — The entire pipeline — from data loading and augmentation through model definition, training, evaluation, and prediction — is consolidated into a single well-documented notebook with detailed markdown explanations for every step.

---

## Architecture Overview

```
                    ┌──────────────────────────────┐
                    │  Brain MRI Image (256×256×3)  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │            U-Net ENCODER                 │
              │                                          │
              │  Block 1: Conv2D(64)×2 → BN → ReLU      │
              │           → MaxPool2D(2×2)               │
              │  Block 2: Conv2D(128)×2 → BN → ReLU     │
              │           → MaxPool2D(2×2)               │
              │  Block 3: Conv2D(256)×2 → BN → ReLU     │
              │           → MaxPool2D(2×2)               │
              │  Block 4: Conv2D(512)×2 → BN → ReLU     │
              │           → MaxPool2D(2×2)               │
              └──────────────────┬──────────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────────┐
              │            BOTTLENECK                     │
              │  Conv2D(1024)×2 → BN → ReLU              │
              └──────────────────┬──────────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────────┐
              │            U-Net DECODER                 │
              │                                          │
              │  Block 6: Conv2DTranspose(512) + Skip    │
              │           → Conv2D(512)×2 → BN → ReLU    │
              │  Block 7: Conv2DTranspose(256) + Skip    │
              │           → Conv2D(256)×2 → BN → ReLU    │
              │  Block 8: Conv2DTranspose(128) + Skip    │
              │           → Conv2D(128)×2 → BN → ReLU    │
              │  Block 9: Conv2DTranspose(64) + Skip     │
              │           → Conv2D(64)×2 → BN → ReLU     │
              └──────────────────┬──────────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────────┐
              │  Conv2D(1, sigmoid) → Tumor Probability  │
              └──────────────────┬──────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌──────────────────┐    ┌──────────────────────┐
         │  Segmentation     │    │  Post-Processing      │
         │  Mask Output      │    │  • Threshold (0.5)    │
         │  (pixel-level)    │    │  • Contour detection  │
         └──────────────────┘    │  • Bounding box draw  │
                                 └──────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | TensorFlow / Keras 3 |
| **Architecture** | U-Net (custom from scratch) |
| **Optimizer** | Adamax (lr=0.001) |
| **Loss Function** | Dice Loss |
| **Metrics** | Dice Coefficient, IoU Coefficient, Accuracy |
| **Data Pipeline** | tf_keras ImageDataGenerator |
| **Image Processing** | OpenCV, scikit-image |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | LGG MRI Segmentation (Kaggle) |
| **Environment** | Kaggle Notebooks (GPU) |

---

## Project Structure

```
brain-mri-segmentation-cv3/
├── README.md                                        # Project documentation
├── requirements.txt                                 # Python dependencies
├── notebooks/
│   └── brain-tumor-segmentation-u-net-80-dice-iou.ipynb  # Complete pipeline
```

---

## How It Works

### 1. Data Loading & Splitting

The dataset is loaded from the LGG MRI Segmentation archive, where each patient folder contains MRI slices and their corresponding tumor masks (files suffixed with `_mask`). A DataFrame is constructed by scanning all mask paths and pairing each one with its corresponding image path (by removing the `_mask` suffix). The dataset is then split into **80% training**, **10% validation**, and **10% test** using `train_test_split` from scikit-learn, ensuring a consistent distribution of tumor-positive and tumor-free slices across all three splits. This stratified approach is critical for reliable evaluation since the proportion of tumorous slices can vary significantly across patients.

### 2. Data Augmentation

Medical imaging datasets are inherently small, which makes neural networks prone to overfitting — memorizing the exact training images instead of learning the general patterns of a tumor. To combat this, training images and their corresponding masks are augmented simultaneously using `tf_keras.preprocessing.image.ImageDataGenerator` with the same random seed, ensuring that the same geometric transformations are applied to both the image and its mask to maintain perfect spatial alignment. Augmentations include rotation (±0.2 radians ≈ 11.5°), width and height shifts (5%), shear (5%), zoom (5%), horizontal flipping, and nearest-neighbor fill mode for any gaps created by the transformations. All images are resized to 256×256 and normalized to the [0, 1] range. A custom generator yields `(image, mask)` pairs for each batch, with masks binarized at the 0.5 threshold to ensure strict binary ground truth.

### 3. U-Net Architecture

The U-Net is built entirely from scratch following the original Ronneberger et al. (2015) design. The **encoder** (contracting path) consists of 4 downsampling blocks, each containing two Conv2D layers with 3×3 kernels and same padding, followed by BatchNormalization and ReLU activation, then MaxPooling2D (2×2) to halve the spatial dimensions. The number of filters doubles at each block: 64 → 128 → 256 → 512, progressively capturing more abstract and complex features. The **bottleneck** at the bottom of the "U" uses two Conv2D layers with 1024 filters and BatchNormalization, serving as the most compressed representation of the input. The **decoder** (expanding path) mirrors the encoder with 4 upsampling blocks, each using Conv2DTranspose (2×2, stride 2) to double the spatial dimensions, followed by concatenation with the corresponding encoder feature map via skip connections. These skip connections are the key innovation of U-Net — they allow the decoder to access fine-grained spatial information from the encoder that would otherwise be lost during downsampling, enabling pixel-precise boundary predictions. Each decoder block concludes with two Conv2D layers and BatchNormalization. The final layer is a Conv2D with a single 1×1 filter and Sigmoid activation, producing a probability map where each pixel value represents the likelihood of belonging to the tumor class.

### 4. Training Strategy

The model is compiled with the **Adamax optimizer** (learning rate 0.001) and **Dice Loss** as the objective function. Dice Loss is the negative of the Dice Coefficient, which directly measures the overlap between the predicted and ground-truth segmentation masks. Unlike cross-entropy loss, which treats each pixel independently, Dice Loss optimizes for the overall shape overlap, making it particularly well-suited for segmentation tasks where the region of interest (the tumor) occupies a small fraction of the total image. Three metrics are tracked during training: pixel-wise Accuracy, IoU Coefficient (Intersection over Union), and Dice Coefficient. Training runs for up to 150 epochs with a batch size of 40, and a `ModelCheckpoint` callback saves the model with the best validation performance (lowest validation loss) to `unet.keras`. This ensures that even if the model begins to overfit in later epochs, the best weights are preserved.

### 5. Evaluation & Visualization

After training completes, a comprehensive evaluation is performed. First, the training history is plotted with four subplots showing the evolution of Accuracy, IoU, Dice Coefficient, and Loss across epochs for both training and validation sets, with the best epoch highlighted. Next, 10 test samples are visualized in a three-column layout: the original MRI slice, the ground truth mask, and the predicted segmentation with a green bounding box drawn around the detected tumor region. Finally, the model is evaluated quantitatively on the train, validation, and test sets, reporting all metrics for each split. An additional prediction section generates 20 random test predictions displayed as triplets: original image, original mask, and binarized model prediction — providing a broad qualitative assessment of the model's segmentation capabilities across diverse tumor presentations.

### 6. Post-Processing: Bounding Box Extraction

The `draw_mask_and_bbox()` function takes an original MRI image and a predicted probability mask, and returns the image with a green bounding box overlaid around each detected tumor region. The process works as follows: the predicted mask is first binarized using a 0.5 threshold, converting the soft probability map into a strict binary mask (0 for background, 1 for tumor). This binary mask is then scaled to 0–255 for OpenCV compatibility, and `cv2.findContours()` with the `RETR_EXTERNAL` flag traces the outer boundaries of all connected tumor regions. For each contour with an area greater than 50 pixels (to filter out noise artifacts), `cv2.boundingRect()` computes the minimum bounding rectangle coordinates `(x, y, width, height)`, and a green rectangle is drawn on a copy of the original MRI. This dual representation — pixel-level segmentation mask plus bounding box localization — makes the output immediately useful for both detailed clinical review and rapid visual scanning by radiologists.

---

## Key Metrics

| Metric | Description |
|--------|------------|
| **Dice Coefficient** | Measures overlap between predicted and ground truth masks: `2 × |A∩B| / (|A|+|B|)`. Primary evaluation metric for medical segmentation. Achieved **≈0.80** |
| **IoU Coefficient** | Intersection over Union: `|A∩B| / |A∪B|`. Standard segmentation benchmark measuring the ratio of overlap to combined area. Achieved **≈0.80** |
| **Accuracy** | Pixel-wise classification accuracy across the entire image. High due to class imbalance (mostly background), so less informative than Dice/IoU |
| **Dice Loss** | Negative Dice Coefficient, used as the training objective to directly optimize segmentation quality rather than pixel-level cross-entropy |

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

Open `notebooks/brain-tumor-segmentation-u-net-80-dice-iou.ipynb` in **Kaggle**, **Google Colab**, or **Jupyter** and update the `data_dir` path:

```python
# For Kaggle:
data_dir = '/kaggle/input/lgg-mri-segmentation/kaggle_3m'

# For local:
data_dir = './kaggle_3m'
```

Then run all cells to train the model and view results. GPU acceleration is recommended for training.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 256 × 256 × 3 |
| Batch Size | 40 |
| Epochs | 150 (with ModelCheckpoint for best weights) |
| Optimizer | Adamax (lr=0.001) |
| Loss Function | Dice Loss |
| Metrics | Accuracy, IoU Coefficient, Dice Coefficient |
| Data Split | 80% Train / 10% Val / 10% Test |
| Data Augmentation | Rotation, shift, shear, zoom, horizontal flip |

---

## Dataset Details

- **Source**: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) by Mateusz Buda
- **Description**: Brain MRI images with manual FLAIR abnormality segmentation masks from The Cancer Imaging Archive (TCIA)
- **Patients**: 110 patients with lower-grade glioma
- **Slices**: ~3,900 MRI slices with paired tumor masks
- **Format**: PNG images (256×256) with `_mask` suffix convention
- **Labels**: Binary segmentation (tumor vs. healthy tissue)

---

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) — Kaggle
- [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)

---

## License

This project is open source and available for educational and personal use. The LGG MRI Segmentation dataset is available through Kaggle and The Cancer Imaging Archive.
