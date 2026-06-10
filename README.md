# Overview
This project explores image classification on the CIFAR-100 dataset, a challenging and widely used dataset containing 100 classes with 600 images each. The work compares multiple approaches, from transfer learning with a pre-trained VGG16 network to designing a custom CNN(Convolutional Neural Network) from scratch, experimenting with different design choices to improve classification accuracy.
The project was completed and run on Google Colab using GPU support.

## Approaches & Results

### Part 1.1 — Feature Extraction with VGG16
VGG16 was used as a fixed feature extractor with custom Dense layers trained on top. Five iterations of the dense architecture were explored, varying the number of layers, activation functions, optimizers, batch sizes, and epochs.

**Best accuracy: 38.26%** — using 3 Dense layers with Adam optimizer and ReLU activation.

---
### Part 1.2 — Fine-Tuning Upper VGG16 Block
The upper convolutional block of VGG16 was unfrozen and retrained alongside a Dense classification layer, improving accuracy to **50%**. Data augmentation was then applied and iteratively adjusted, achieving a further improvement to **51.2%**, confirming that data augmentation helps reduce overfitting and improves generalisation.

---
### Part 1.3 — Full VGG16 Network Training
All layers of VGG16 were made trainable and retrained with a lower learning rate. Initial training for 10 epochs achieved **55.8%** accuracy. After increasing epochs to 20 and reducing batch size to 128, accuracy improved further to **59.6%**. Training the full network allowed end-to-end feature learning and task-specific adaptation across all convolutional blocks.

---
### Part 1.4 — Custom Convolutional Neural Network from Scratch
A Convolutional Neural Network was designed and built entirely from scratch, going through multiple iterations:

- **Baseline model** — Basic CNN with convolutional blocks, max pooling, dense layers, dropout regularisation, and early stopping. Initial accuracy: **23%**
- **Iteration 2** — Increased model depth, added more filters, residual connections, and model ensembling. Improved accuracy: **32%**
- **Iteration 3 — Simplified model** — Reduced number of layers and parameters to address overfitting. Accuracy dropped to **25%**, confirming a more sophisticated architecture performs better on this dataset
- **Final best model** — Multiple convolutional layers with batch normalisation, max pooling, dense layers, Adam optimiser, early stopping, learning rate reduction callbacks, and data augmentation. Best accuracy: **32.4%**
100 classes | 600 images per class | 32x32 pixel images

## Key Observations

- Full VGG16 fine-tuning significantly outperformed feature extraction alone (59.6% vs 38.26%)
- Data augmentation provided consistent improvements across fine-tuning experiments
- For the custom Convolutional Neural Network, deeper architectures with batch normalisation and residual connections outperformed simpler models
- Lower learning rates and larger batch sizes were critical for stable full network training
- CIFAR-100 is a particularly challenging dataset — even well-optimised models face accuracy limitations due to the large number of classes and small image size

---

## Dataset
CIFAR-100 — available via [Keras datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100)
- 100 classes
- 600 images per class
- 32x32 pixel images

---
## Tools & Libraries
- **Language:** Python
- **Framework:** TensorFlow, Keras
- **Environment:** Google Colab (GPU)

---
