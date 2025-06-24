# High-Resolution VAE-GAN for Medical Image Translation

This project implements a VAE-GAN (Variational Autoencoder - Generative Adversarial Network) model for translating between MRI and CT scan images, with additional super-resolution capabilities. The model can generate high-quality CT scans from MRI images and vice versa, while maintaining important medical features.

---

## Abstract

The study focuses on generating CT images from MRI images using unsupervised learning with VAE-CycleGAN. Due to the limited number of samples in the dataset, probabilistic models were employed to form the latent space. The model achieves good results when transitioning from MRI to CT images and acceptable results for the reverse direction.

---

## Key Features

- Bidirectional translation between MRI and CT scan images
- Probabilistic latent space modeling using VAE
- Multi-scale discriminator architecture
- Data augmentation for improved training
- Super-resolution capabilities using RRDB-Net
- Comprehensive evaluation metrics (PSNR, SSIM, AUC-ROC, F1-Score)

---

## Model Architecture

### Generator (U-Net)
- Encoder-decoder architecture with skip connections
- VAE integration for probabilistic latent space
- Group normalization and residual connections
- Multiple convolutional layers with varying filter sizes

### Discriminator
- Multi-scale patch-based discrimination
- Derivative independence methodology
- Deep feature extraction capabilities
- Multiple output scales for better feature learning

---

## Requirements

```bash
tensorflow>=2.0.0
torch>=1.7.0
numpy
opencv-python
scikit-learn
scikit-image
matplotlib
```

---

## Dataset

The model uses the CT-to-MRI CGAN dataset from Kaggle. The dataset contains paired MRI and CT scan images for training and testing.

---

## Training

The model was trained for 50,000 epochs with the following parameters:
- Batch size: 4 (8 after augmentation)
- Learning rate: 0.0001
- Weight decay: 6e-8
- Image size: 256x256x3

---

## Evaluation Metrics

The model is evaluated using multiple metrics:
- Binary Cross-Entropy Loss
- AUC-ROC Score
- F1-Score
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

---

## Results

The model achieves:
- Good performance in MRI to CT translation
- Acceptable performance in CT to MRI translation
- High-quality super-resolution output
- Preservation of important medical features

---

## Usage

1. Clone the repository
2. Install dependencies
3. Download the dataset
4. Run the training script
5. Use the trained model for inference

---

## Model Weights

The trained model weights are saved in the following format:
- Generator (MRI to CT): `g_target_weights.h5`
- Generator (CT to MRI): `g_source_weights.h5`
- Discriminator (MRI): `d_source_weights.h5`
- Discriminator (CT): `d_target_weights.h5`

---

## Limitations

- Performance is affected by the limited dataset size
- CT to MRI translation shows higher loss values
- Training requires significant computational resources

---

## License

This project is licensed under the MIT License 

---

## Author 

Sunil Kumawat

