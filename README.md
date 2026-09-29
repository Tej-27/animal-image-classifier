# 🐾 AI Animal Vision — Animal Image Classifier

An end-to-end Machine Learning image classification project that classifies animal images into three categories:

- 🐱 Cat
- 🐶 Dog
- 🦁 Wild

The project uses a **Random Forest Classifier** trained on the AFHQ (Animal Faces-HQ) dataset and is deployed as an interactive web application using **Streamlit**.

---

## 🚀 Live Demo

🔗 **Live Application:**  
[AI Animal Vision](https://ai-animal-vision.streamlit.app/)

> The live link will be available after deploying the application on Streamlit Community Cloud.

---

## 📌 Project Overview

The goal of this project is to build a machine learning model capable of classifying animal images into three categories.

The complete workflow includes:

1. Dataset collection
2. Image preprocessing
3. Image resizing
4. Feature extraction
5. Pixel normalization
6. Model training
7. Hyperparameter tuning
8. Model evaluation
9. Model compression
10. Web application development
11. Deployment

---

## 📊 Dataset

The project uses the **Animal Faces-HQ (AFHQ)** dataset.

### Classes

| Class | Description |
|-------|-------------|
| 🐱 Cat | Domestic cat images |
| 🐶 Dog | Domestic dog images |
| 🦁 Wild | Wild animal images |

### Dataset Distribution

| Dataset | Cat | Dog | Wild | Total |
|---------|-----|-----|------|-------|
| Training | 5,153 | 4,739 | 4,738 | 14,630 |
| Validation | 500 | 500 | 500 | 1,500 |

---

## 🧠 Machine Learning Model

The final model uses a:

**Random Forest Classifier**

### Model Configuration

```text
Algorithm: Random Forest
Number of Trees: 300
Maximum Depth: None
Minimum Samples per Leaf: 1
Random State: 42
