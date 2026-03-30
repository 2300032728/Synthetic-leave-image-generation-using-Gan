<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/fdba9e7e-f372-431a-88df-76702e8a7f88" />Synthetic Diseased Leaf Image Generation using DCGAN
Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic images of diseased plant leaves. The primary objective is to address data scarcity in agricultural datasets by producing realistic artificial samples that can be used for training and research purposes.

Problem Statement

Plant disease detection systems rely heavily on large and diverse datasets. However, collecting such datasets is difficult, time-consuming, and often results in class imbalance. This limitation reduces the performance and generalization capability of deep learning models.

Proposed Solution

This project develops a DCGAN-based generative model that learns the distribution of diseased leaf images and generates new synthetic samples. These generated images can be used to augment datasets and improve model robustness.

Methodology
Data Collection
A dataset of diseased plant leaf images is collected from available sources.
Preprocessing
Images are resized to a fixed dimension (e.g., 64×64 or 128×128) and normalized to a suitable range for training.
Model Architecture
Generator: Uses transposed convolution layers to generate images from random noise vectors.
Discriminator: Uses convolutional layers to classify images as real or fake.
Training
The generator and discriminator are trained simultaneously in an adversarial setup. The generator aims to produce realistic images, while the discriminator learns to distinguish them from real images.
Evaluation
Generated images are evaluated visually and can optionally be assessed using metrics such as Fréchet Inception Distance (FID).
Tech Stack
Programming Language: Python
Deep Learning Framework: PyTorch
Libraries: NumPy, OpenCV/PIL, Matplotlib
Environment: Google Colab / Local GPU
Project Structure

project-root/
│── data/ # Raw dataset
│── preprocessed/ # Processed images
│── checkpoints/ # Saved model weights
│── samples/ # Generated outputs
│── generator.py # Generator model
│── discriminator.py # Discriminator model
│── train.py # Training script
│── data_loader.py # Data pipeline
│── README.md

Training Configuration

Image Size: 64×64 / 128×128
Latent Vector Size: 100
Batch Size: 64
Learning Rate: 0.0002
Optimizer: Adam
Beta1: 0.5
Epochs: 100–500

Execution
Install dependencies
pip install torch torchvision numpy matplotlib opencv-python pillow
Run training
python train.py

Generated images will be stored in the samples directory.

Results

The DCGAN model is capable of generating synthetic diseased leaf images that resemble real data. These images can be used to improve dataset size and diversity, leading to better performance in disease classification models.

Applications
Plant disease detection systems
Dataset augmentation for deep learning models
Research in precision agriculture
Improving robustness of image classification models
Limitations
Generated images may lack fine-grained details
Evaluation is primarily visual unless advanced metrics are used
Performance depends on dataset quality and size
Future Work
Improve image quality using advanced GAN variants such as WGAN or StyleGAN
Apply quantitative evaluation metrics like FID and IS
Integrate generated data into classification pipelines
Develop a web interface for real-time image generation
Conclusion

This project demonstrates the effectiveness of DCGAN in generating synthetic diseased leaf images. It provides a scalable solution for addressing data limitations in agricultural AI and supports the development of more robust plant disease detection systems.

