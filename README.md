# CIFAR-100 Image Classification with VGG16 and Custom CNN

## Overview
This project explores image classification on the CIFAR-100 dataset, a challenging and widely used dataset containing 100 classes with 600 images each. The work compares multiple approaches, from transfer learning with a pre-trained VGG16 network to designing a custom Convolutional Neural Network (CNN) from scratch, experimenting with different design choices to improve classification accuracy.

The project was completed and run on Google Colab using GPU support.

---

## Approaches & Results

### Part 1.1 — Transfer Learning with VGG16

#### Feature Extraction
VGG16 was used as a fixed feature extractor with custom Dense layers trained on top. Three iterations of the architecture were explored, adjusting the number of layers, dropout, batch normalisation, and early stopping.

| Iteration | Key Changes | Test Accuracy |
|---|---|---|
| Baseline | 3 Dense layers, Adam, 15 epochs | 38.29% |
| Iteration 1 | Deeper architecture, 30 epochs | 38.24% |
| Iteration 2 | Batch normalisation, increased dropout, early stopping | 43.72% |
| Iteration 3 | Reduced epochs to 10 | 42.60% |

**Best accuracy: 43.72%** — achieved with batch normalisation, dropout regularisation, and early stopping.

---

#### Fine-Tuning Upper VGG16 Block
The top four layers of VGG16 were unfrozen and retrained alongside custom Dense layers, using a low learning rate (1e-5) with RMSprop. Data augmentation was then applied and iteratively adjusted.

| Iteration | Key Changes | Test Accuracy |
|---|---|---|
| Fine-tuning only | Top 4 layers unfrozen, RMSprop (1e-5) | 48.31% |
| With data augmentation | Width/height shift, horizontal flip | 46.70% |
| Enhanced augmentation | Added rotation, shear, zoom, fewer dense layers | 53.70% |

**Best accuracy: 53.70%** — achieved with enhanced data augmentation and a simplified dense architecture.

**On data augmentation:** Without data augmentation the accuracy was 48.3%. With enhanced data augmentation it reached 53.7% — an improvement of 5.4 percentage points. This confirms that data augmentation helps the model generalise better by reducing overfitting and exposing it to more diverse training examples.

---

### Full VGG16 Network Training
All layers of VGG16 were made trainable and retrained with a low learning rate, comprehensive data augmentation, and batch normalisation.

| Iteration | Key Changes | Test Accuracy |
|---|---|---|
| Full network training | All layers trainable, RMSprop (1e-5), batch size 128 | 58.90% |
| Modified full network | Adam optimizer, higher learning rate, added dropout | 54.90% |

**Best accuracy: 58.90%** — achieved with RMSprop, full data augmentation, and batch normalisation.

---

### Part 1.2 — Custom Convolutional Neural Network from Scratch
A Convolutional Neural Network was designed and built entirely from scratch, going through multiple iterations:

| Iteration | Key Changes | Test Accuracy |
|---|---|---|
| Baseline CNN | 3 conv blocks, batch norm, max pooling, dropout | 39.60% |
| Iteration 2 | SeparableConv2D, data augmentation, early stopping, ReduceLROnPlateau | 35.70% |
| Final model | Standard Conv2D, deeper network, L2 regularisation, progressive dropout | 44.30% |

**Best accuracy: 44.30%** — achieved with a deeper architecture, L2 regularisation, progressive dropout (0.3, 0.4, 0.5), and extended training up to 50 epochs.

Key design decisions in the final model:
- Two Conv2D layers per convolutional block for increased depth
- L2 regularisation on all convolutional and dense layers to prevent overfitting
- Progressive dropout rates across blocks for balanced regularisation
- Data augmentation with rotation, shift, shear, zoom, and horizontal flip
- Early stopping and learning rate reduction on plateau callbacks

---

## Overall Results Summary

| Approach | Best Test Accuracy |
|---|---|
| VGG16 Feature Extraction | 43.72% |
| VGG16 Fine-tuning (upper block) | 53.70% |
| VGG16 Full Network Training | 58.90% |
| Custom CNN from Scratch | 44.30% |

**Best overall accuracy: 58.90%** — achieved by training the full VGG16 network with comprehensive data augmentation and careful optimisation.

---

## Key Observations
- Full VGG16 training consistently outperformed feature extraction and partial fine-tuning
- Data augmentation provided consistent improvements across all fine-tuning experiments
- For the custom CNN, deeper architectures with L2 regularisation and progressive dropout outperformed simpler models
- Lower learning rates were essential for stable full network training
- SeparableConv2D did not outperform standard Conv2D layers on this dataset
- CIFAR-100 is a particularly challenging dataset due to its large number of classes and small image size

---

## Dataset
[CIFAR-100](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100) — available via Keras datasets
- 100 classes
- 600 images per class
- 32x32 pixel images

---

## Tools & Libraries
- **Language:** Python
- **Framework:** TensorFlow, Keras
- **Environment:** Google Colab (GPU)

---
