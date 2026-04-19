# 🌿 Synthetic Diseased Leaf Image Generation using DCGAN

<p align="center">
  <img src="https://github.com/user-attachments/assets/fdba9e7e-f372-431a-88df-76702e8a7f88" width="800"/>
  <br>
  <em>DCGAN Architecture for Diseased Leaf Image Generation</em>
</p>

---

## 📌 Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate synthetic images of diseased plant leaves.

The objective is to solve **data scarcity and class imbalance** in agricultural datasets by generating realistic artificial samples for training and research.

---

## ❗ Problem Statement

Plant disease detection systems require:

* Large datasets
* Diverse image samples
* Balanced class distribution

However:

* Data collection is **time-consuming**
* Datasets are often **imbalanced**
* Models show **poor generalization**

---

## 💡 Proposed Solution

This project develops a **DCGAN-based model** that:

* Learns patterns from real diseased leaf images
* Generates **synthetic samples**
* Improves dataset **size and diversity**
* Enhances model **robustness**

---

## ⚙️ Methodology

### 📥 Data Collection

Collect diseased leaf images from available datasets.

### 🔄 Preprocessing

* Resize images (64×64 / 128×128)
* Normalize pixel values

### 🧠 Model Architecture

* **Generator:** Transposed convolution layers generate images from noise
* **Discriminator:** Convolution layers classify real vs fake images

### 🔁 Training

* Generator tries to fool the discriminator
* Discriminator learns to detect fake images
* Both improve through adversarial learning

### 📊 Evaluation

* Visual inspection
* Optional metric: **FID (Fréchet Inception Distance)**

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **NumPy**
* **OpenCV / PIL**
* **Matplotlib**
* **Google Colab / Local GPU**
* **Git & GitHub**

---

## 📁 Project Structure

```bash id="b2l1kt"
project-root/
│── data/
│── preprocessed/
│── checkpoints/
│── samples/
│── generator.py
│── discriminator.py
│── train.py
│── data_loader.py
│── README.md
```

---

## ⚙️ Training Configuration

* Image Size: 64×64 / 128×128
* Latent Vector Size: 100
* Batch Size: 64
* Learning Rate: 0.0002
* Optimizer: Adam
* Beta1: 0.5
* Epochs: 100–500

---

## 🚀 Installation & Execution

### 1️⃣ Clone the Repository

```bash id="r0v0h2"
git clone https://github.com/2300032728/Synthetic-leave-image-generation-using-Gan.git
cd Synthetic-leave-image-generation-using-Gan
```

---

### 2️⃣ Install Dependencies

```bash id="kqf7s1"
pip install torch torchvision numpy matplotlib opencv-python pillow
```

---

### 3️⃣ Run Training

```bash id="2h4y4l"
python train.py
```

👉 Generated images will be stored in the **samples/** directory.

---

## 📊 Results

* Generates realistic diseased leaf images
* Improves dataset diversity
* Enhances model performance

---

## 🌱 Applications

* Plant disease detection
* Dataset augmentation
* Precision agriculture
* Image classification improvement

---

## ⚠️ Limitations

* Fine details may be limited
* Evaluation is mostly visual
* Performance depends on dataset quality

---

## 🔮 Future Work

* Improve using **WGAN / StyleGAN**
* Apply metrics like **FID, IS**
* Integrate with classification models
* Build a web interface

---

## 🏁 Conclusion

This project demonstrates how **DCGAN can generate synthetic diseased leaf images**, helping overcome dataset limitations and improving agricultural AI systems.

---

## 👤 Author

**Korukonda Shyamala**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
