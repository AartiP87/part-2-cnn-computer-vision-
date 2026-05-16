# Part 2: CNN Computer Vision — Manufacturing Defect Classification

## Overview

This project builds a Convolutional Neural Network (CNN) prototype to automatically classify product surface images into one of four categories: **normal**, **scratch**, **dent**, or **stain**. It demonstrates how CNNs can power automated visual quality inspection in manufacturing.

---

## Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── images/
│   ├── normal/
│   ├── scratch/
│   ├── dent/
│   └── stain/
├── labels.csv
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── class_distribution.png
    ├── sample_images.png
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png
```

---

## Task 1: Problem Identification

**Problem Type: Image Classification**

The dataset represents a **multi-class image classification** problem. Each image must be assigned exactly one label from four categories. This is distinct from:
- **Object detection** — no bounding boxes needed
- **Semantic segmentation** — no pixel-level masks
- **Instance segmentation** — no need to distinguish multiple defect instances

The model takes a full image as input and outputs a probability score for each of the 4 defect classes.

---

## Task 2: Dataset Exploration

| Property | Value |
|----------|-------|
| Number of classes | 4 (normal, scratch, dent, stain) |
| Images per class | 120 |
| Total images | 480 |
| Image dimensions | 96 × 96 px (resized to 64×64 for training) |
| Color mode | RGB |
| Class balance | Perfectly balanced — no imbalance |

Since the dataset is perfectly balanced, no oversampling or class weighting is required.

---

## Task 3: Image Preprocessing

| Step | Detail |
|------|--------|
| Resizing | All images resized to 64×64 px |
| Normalization | Pixel values scaled from [0, 255] → [0.0, 1.0] |
| Split | 70% train / 15% validation / 15% test (stratified) |
| Augmentation | Not applied — balanced synthetic dataset |

Augmentation (horizontal flip, rotation, brightness jitter) would be applied for real-world datasets that are imbalanced or have limited training samples.

---

## Task 4: CNN Model Architecture

We implement the CNN pipeline manually using NumPy, then connect it to a dense classifier:

```
Input Image (64×64×3)
        │
 [Convolution Layer]  ← 5 filters: Sobel X/Y, Laplacian, Blur, Sharpen
        │
 [ReLU Activation]    ← max(0, x) — zeroes out negative values
        │
 [Max Pooling 2×2]    ← stride 2, reduces map from 64×64 to 32×32
        │
 [Flatten Layer]      ← global stats (mean, max, std) per pooled map
        │
 [Dense Layer]        ← StandardScaler + SVC (RBF kernel)
        │
 [Output Layer]       ← predict_proba over 4 classes (softmax-equivalent)
```

---

## Task 5: Model Training and Evaluation

| Metric | Value |
|--------|-------|
| Train Accuracy | 100% |
| Validation Accuracy | 100% |
| Test Accuracy | 100% |
| Precision (macro avg) | 1.00 |
| Recall (macro avg) | 1.00 |
| F1-Score (macro avg) | 1.00 |

> **Note:** 100% accuracy is expected here because this is a **synthetic dataset** with very visually distinct patterns per class (solid color textures, clear geometric defects). Real-world defect images would be significantly more challenging.

The confusion matrix (saved to `results/confusion_matrix.png`) shows perfect classification across all four classes with no misclassifications on the test set.

---

## Task 6: CNN Concept Explanation

### What is Convolution?

Convolution is the core operation of a CNN. A small matrix of weights called a **filter** (or kernel) slides across the input image. At each position, it computes the dot product between its weights and the image patch beneath it. The result is a **feature map** — a 2D representation highlighting the patterns that filter is sensitive to (e.g., edges, corners, textures).

For example:
- A horizontal Sobel filter produces a feature map that lights up wherever there are horizontal edges.
- A Laplacian filter detects rapid intensity changes — useful for finding scratches or dents.

### Why is Pooling Used?

Pooling reduces the spatial dimensions of feature maps. **Max pooling** takes the maximum value in each non-overlapping region (e.g., 2×2 blocks). This achieves two things:

1. **Dimensionality reduction:** Smaller maps → fewer parameters → less computation and overfitting.
2. **Translation invariance:** If a feature (e.g., a scratch) shifts slightly in the image, max pooling still detects it in the same pooled region. The model becomes robust to minor positional variations.

### Why is ReLU Commonly Used in CNNs?

ReLU (Rectified Linear Unit) is defined as `f(x) = max(0, x)`. It:

- **Introduces non-linearity:** Without activation functions, stacking convolution layers is mathematically equivalent to a single linear operation. ReLU allows the network to learn complex non-linear relationships.
- **Avoids the vanishing gradient problem:** Sigmoid and tanh functions saturate (output values near 0 or 1), causing gradients to vanish during backpropagation. ReLU's gradient is simply 1 for positive inputs, keeping gradients flowing through deep networks.
- **Is computationally cheap:** Just a threshold operation — very fast to compute.

### Why Are CNNs Better Than Feed-Forward Networks for Image Data?

| Property | Feed-Forward Network (FFN) | CNN |
|----------|---------------------------|-----|
| **Spatial structure** | Flattens image → loses all spatial relationships | Preserves 2D structure via convolution |
| **Parameter count** | Each pixel connected to every neuron → millions of parameters even for small images | Filters are shared across all positions → far fewer parameters |
| **Feature learning** | No notion of local patterns | Learns hierarchical local features (edges → shapes → objects) |
| **Translation invariance** | None — pixel at position (10,10) is independent of (11,10) | Pooling provides invariance to small positional shifts |
| **Scalability** | Breaks down for high-resolution images | Scales well due to parameter sharing |

CNNs exploit two key properties of images: **locality** (nearby pixels are related) and **stationarity** (the same pattern can appear anywhere in the image). FFNs treat every pixel as an independent, unrelated feature.

---

## Task 7: Business Use Case Mapping

### Domain: Manufacturing — Automated Visual Quality Inspection

**Problem Statement:**  
In high-speed production lines (automotive parts, consumer electronics, food packaging, pharmaceuticals), products must be inspected for surface defects before shipping. Manual inspection by human operators is slow, inconsistent, and does not scale with production volume.

**How This CNN Solution Applies:**

1. **Image capture:** Cameras mounted on conveyor belts photograph each product unit as it passes.
2. **Real-time inference:** The CNN classifies each image in milliseconds as `normal`, `scratch`, `dent`, or `stain`.
3. **Automated diversion:** A pneumatic actuator physically removes flagged units from the production line before packaging.
4. **Defect logging:** Each classification — label, confidence score, timestamp, production shift — is written to a database for quality reporting and root-cause analysis.

**Business Impact:**

| Benefit | Description |
|---------|-------------|
| Cost reduction | Catches defects before packaging, reducing scrap and rework |
| Throughput | Inspection keeps pace with the production line — no bottleneck |
| Consistency | No human fatigue or subjective judgment |
| Traceability | Every unit has a logged classification for auditing |
| Process improvement | Defect trend analysis helps identify root causes in production |

**Real-World Deployment Context:**  
Solutions of this type are deployed in automotive stamping plants (detecting panel dents), PCB manufacturing (detecting solder defects), food processing (detecting contamination or shape anomalies), and pharmaceutical packaging (detecting broken seals or label misalignment).

---

## Setup and Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook notebook.ipynb
```

Execute all cells in order. The notebook will:
1. Load and explore the dataset
2. Preprocess images
3. Extract CNN-style features
4. Train and evaluate the model
5. Save all result plots to `results/` and `sample_predictions/`

---

## Results

| Output | Location |
|--------|----------|
| Class distribution chart | `results/class_distribution.png` |
| Sample class images | `results/sample_images.png` |
| Training/validation curves | `results/accuracy_loss_curves.png` |
| Confusion matrix | `results/confusion_matrix.png` |
| Sample predictions | `sample_predictions/prediction_outputs.png` |
