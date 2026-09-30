# 🐾 AI Animal Vision — Animal Image Classification

<p align="center">
  <b>End-to-End Machine Learning Image Classification Project</b>
</p>

<p align="center">
  A Machine Learning application that classifies animal images into
  <b>Cat</b>, <b>Dog</b>, or <b>Wild</b> using a Random Forest Classifier.
</p>

<p align="center">

🚀 <a href="https://animalimageclassifier.streamlit.app"> <b>Live Demo</b> </a>

</p>

---

## 🌐 Live Application

### 🚀 Try the application

**[Open AI Animal Vision →](https://animalimageclassifier.streamlit.app)**

The application allows users to upload an animal image and receive:

* 🐱 Predicted animal class
* 🎯 Prediction confidence
* 📊 Probability for each class
* 🖼️ Uploaded image preview
* ℹ️ Model information

The application is deployed using **Streamlit Community Cloud**.

---

# 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Project Objectives](#-project-objectives)
* [Dataset](#-dataset)
* [Dataset Distribution](#-dataset-distribution)
* [Machine Learning Workflow](#-machine-learning-workflow)
* [Image Preprocessing](#-image-preprocessing)
* [Feature Extraction](#-feature-extraction)
* [Machine Learning Models](#-machine-learning-models)
* [Model Experiments](#-model-experiments)
* [Hyperparameter Tuning](#-hyperparameter-tuning)
* [Image Resolution Experiment](#-image-resolution-experiment)
* [Final Model](#-final-model)
* [Model Evaluation](#-model-evaluation)
* [Web Application](#-web-application)
* [Application Workflow](#-application-workflow)
* [Model Serialization](#-model-serialization)
* [Project Structure](#-project-structure)
* [Technologies Used](#-technologies-used)
* [Installation](#-installation)
* [Running the Application](#-running-the-application)
* [Deployment](#-deployment)
* [Limitations](#-limitations)
* [Future Improvements](#-future-improvements)
* [Learning Outcomes](#-learning-outcomes)
* [Key Project Statistics](#-key-project-statistics)
* [Conclusion](#-conclusion)
* [Author](#-author)

---

# 🚀 Project Overview

**AI Animal Vision** is an end-to-end Machine Learning image classification project developed to classify animal images into three categories:

| Category | Class |
| -------- | ----- |
| 🐱       | Cat   |
| 🐶       | Dog   |
| 🦁       | Wild  |

The project covers the complete Machine Learning lifecycle:

```text
Dataset
   ↓
Data Collection
   ↓
Image Preprocessing
   ↓
Feature Extraction
   ↓
Data Normalization
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Application
   ↓
Cloud Deployment
```

The final model is a **Random Forest Classifier with 300 trees**.

It achieved approximately:

# **82.60% Validation Accuracy**

The trained model was then integrated into a Streamlit web application and deployed publicly.

---

# 🎯 Problem Statement

Image classification is a supervised Machine Learning problem where an algorithm learns to associate visual patterns with predefined categories.

The objective of this project is to build a system that takes an animal image as input and predicts which of the following categories it belongs to:

```text
Input Image
     ↓
Machine Learning Model
     ↓
┌───────────────┐
│ Cat           │
│ Dog           │
│ Wild          │
└───────────────┘
```

The project focuses on demonstrating an end-to-end Machine Learning pipeline using traditional Machine Learning techniques rather than a deep learning architecture.

---

# 🎯 Project Objectives

The major objectives of the project are:

1. Collect and prepare an image classification dataset.
2. Convert image data into numerical features.
3. Resize images to a consistent resolution.
4. Normalize image pixel values.
5. Train multiple Machine Learning models.
6. Compare model performance.
7. Tune Random Forest hyperparameters.
8. Experiment with different image resolutions.
9. Evaluate the final model.
10. Serialize the trained model.
11. Build an interactive Streamlit application.
12. Deploy the application to the cloud.
13. Provide users with prediction probabilities and confidence information.

---

# 📊 Dataset

The project uses the **Animal Faces-HQ (AFHQ)** dataset.

The dataset contains animal face images organized into different categories.

For this project, the following three categories were used:

* 🐱 Cat
* 🐶 Dog
* 🦁 Wild

---

# 📁 Dataset Structure

The dataset follows a structure similar to:

```text
afhq/
│
├── train/
│   ├── cat/
│   ├── dog/
│   └── wild/
│
└── val/
    ├── cat/
    ├── dog/
    └── wild/
```

Each class contains images belonging to that category.

---

# 📈 Dataset Distribution

## Training Dataset

| Class     | Number of Images |
| --------- | ---------------: |
| 🐱 Cat    |            5,153 |
| 🐶 Dog    |            4,739 |
| 🦁 Wild   |            4,738 |
| **Total** |       **14,630** |

## Validation Dataset

| Class     | Number of Images |
| --------- | ---------------: |
| 🐱 Cat    |              500 |
| 🐶 Dog    |              500 |
| 🦁 Wild   |              500 |
| **Total** |        **1,500** |

### Total Images Used

```text
Training Images   : 14,630
Validation Images : 1,500
Total Images      : 16,130
Classes           : 3
```

---

# 🔄 Machine Learning Workflow

The complete workflow used in this project is:

```text
Animal Images
      │
      ▼
Image Loading
      │
      ▼
RGB Conversion
      │
      ▼
Image Resizing
      │
      ▼
NumPy Conversion
      │
      ▼
Image Flattening
      │
      ▼
Pixel Normalization
      │
      ▼
Machine Learning Models
      │
      ├── Linear SVM
      │
      └── Random Forest
               │
               ▼
       Hyperparameter Tuning
               │
               ▼
         Final Random Forest
               │
               ▼
       Model Evaluation
               │
               ▼
         Model Serialization
               │
               ▼
       Streamlit Application
               │
               ▼
        Cloud Deployment
```

---

# 🖼️ Image Preprocessing

Machine Learning algorithms cannot directly work with image files such as JPG or PNG.

Therefore, every image is transformed into numerical data.

The preprocessing pipeline consists of:

```text
Original Image
      ↓
Convert to RGB
      ↓
Resize to 64 × 64
      ↓
Convert to NumPy Array
      ↓
Flatten
      ↓
Normalize Pixel Values
```

---

## 1. RGB Conversion

Every image is converted into RGB format.

```python
image = Image.open(image_path).convert("RGB")
```

This ensures that each image contains exactly three color channels:

```text
Red
Green
Blue
```

---

## 2. Image Resizing

All images are resized to:

```text
64 × 64 pixels
```

using:

```python
image = image.resize((64, 64))
```

Using a fixed resolution ensures that every image produces the same number of features.

---

## 3. NumPy Conversion

The processed image is converted into a NumPy array:

```python
image_array = np.array(image)
```

The resulting shape is:

```text
64 × 64 × 3
```

---

# 🔢 Feature Extraction

After resizing, each image is flattened into a one-dimensional feature vector.

```python
image_flat = image_array.reshape(1, -1)
```

The number of features is:

```text
64 × 64 × 3
```

Therefore:

```text
12,288 features per image
```

Each feature represents a pixel-channel value.

---

# 📏 Pixel Normalization

Raw image pixel values range between:

```text
0 – 255
```

These values are normalized to:

```text
0 – 1
```

using:

```python
image_flat = image_flat.astype("float32") / 255.0
```

This exact preprocessing pipeline is also used in the deployed Streamlit application.

Maintaining the same preprocessing during training and inference is important because the deployed model expects the same feature representation that it learned during training.

---

# 🤖 Machine Learning Models

Several Machine Learning approaches were evaluated during development.

The main approaches included:

### 1. Linear Support Vector Machine

A Linear SVM was tested using the flattened pixel features.

Approximate validation accuracy:

```text
74%
```

---

### 2. Random Forest

Random Forest produced better performance on the selected dataset.

Multiple Random Forest configurations were tested to identify a suitable combination of parameters.

---

# 🧪 Model Experiments

## Random Forest — Number of Trees

Different values of `n_estimators` were tested.

| Number of Trees | Validation Accuracy |
| --------------: | ------------------: |
|             100 |              81.40% |
|             200 |              81.93% |
|         **300** |          **82.60%** |
|             500 |              81.67% |

The best result among the tested configurations was obtained using:

```python
n_estimators=300
```

---

# 🌲 Hyperparameter Tuning

Additional Random Forest parameters were evaluated.

---

## Maximum Depth

| `max_depth` |   Accuracy |
| ----------: | ---------: |
|          10 |     80.20% |
|          20 |     81.73% |
|          30 |     82.07% |
|          40 |     82.33% |
|        None | **82.60%** |

The final model uses:

```python
max_depth=None
```

---

## Minimum Samples per Leaf

| `min_samples_leaf` |   Accuracy |
| -----------------: | ---------: |
|                  1 | **82.60%** |
|                  2 |     81.00% |
|                  4 |     81.33% |
|                  8 |     81.07% |
|                 16 |     79.93% |

The final model uses:

```python
min_samples_leaf=1
```

---

# 🖼️ Image Resolution Experiment

Image resolution was also investigated to determine whether higher-resolution images would improve model performance.

---

## 64 × 64

Number of features:

```text
64 × 64 × 3
= 12,288
```

Validation accuracy:

```text
82.60%
```

---

## 128 × 128

Number of features:

```text
128 × 128 × 3
= 49,152
```

Validation accuracy:

```text
≈82%
```

Increasing the image resolution did not provide a meaningful improvement in validation accuracy.

Therefore, the final application uses:

```text
64 × 64
```

This provides a smaller feature space while maintaining comparable performance.

---

# 🏆 Final Model

The final model is a Random Forest Classifier configured as:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
```

### Final Configuration

| Parameter                | Value         |
| ------------------------ | ------------- |
| Algorithm                | Random Forest |
| Number of Trees          | 300           |
| Maximum Depth            | None          |
| Minimum Samples per Leaf | 1             |
| Random State             | 42            |
| Input Resolution         | 64 × 64       |
| Color Format             | RGB           |
| Features                 | 12,288        |

---

# 📈 Model Evaluation

The final model achieved:

# **82.60% Validation Accuracy**

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

---

## Classification Report

| Class       | Precision |   Recall | F1-Score |   Support |
| ----------- | --------: | -------: | -------: | --------: |
| Cat         |      0.83 |     0.80 |     0.81 |       500 |
| Dog         |      0.83 |     0.82 |     0.83 |       500 |
| Wild        |      0.82 |     0.86 |     0.84 |       500 |
| **Overall** |  **0.83** | **0.83** | **0.83** | **1,500** |

---

# 📊 Understanding the Evaluation Metrics

## Accuracy

Accuracy represents the percentage of validation images correctly classified.

The final model achieved:

```text
82.60%
```

---

## Precision

Precision represents the proportion of images predicted as a particular class that actually belong to that class.

---

## Recall

Recall represents the proportion of actual images belonging to a class that were correctly identified.

---

## F1-Score

F1-score provides a balance between precision and recall.

---

# 🌐 Streamlit Web Application

The trained model was integrated into an interactive Streamlit application.

### Live Application

🚀 **https://animalimageclassifier.streamlit.app**

The application provides a user-friendly interface for making predictions without requiring users to run Python code.

---

# ✨ Application Features

## 📤 Image Upload

Users can upload:

```text
JPG
JPEG
PNG
WEBP
```

images.

---

## 🖼️ Image Preview

The uploaded image is displayed in the application before displaying the prediction.

---

## 🤖 Animal Classification

The model predicts one of:

```text
🐱 CAT
🐶 DOG
🦁 WILD
```

---

## 🎯 Prediction Confidence

The application displays the highest predicted probability as the prediction confidence.

For example:

```text
Prediction: DOG

Confidence: 91.42%
```

---

## 📊 Class Probabilities

The application also displays the probabilities for all three classes.

Example:

```text
🐱 Cat  — 4.21%
🐶 Dog  — 91.42%
🦁 Wild — 4.37%
```

This provides more information than simply displaying the predicted class.

---

## 🎨 User Interface

The application includes:

* Professional dark theme
* Responsive layout
* Image upload component
* Image preview
* Prediction card
* Confidence display
* Probability bars
* Model statistics
* Sidebar information
* Supported class information
* Prediction explanation
* Footer

---

# 🔄 Application Prediction Pipeline

When a user uploads an image, the application follows this process:

```text
                  User Upload
                       │
                       ▼
                Read Image
                       │
                       ▼
                Convert RGB
                       │
                       ▼
                 Resize 64×64
                       │
                       ▼
              Convert to NumPy
                       │
                       ▼
                Flatten Image
                       │
                       ▼
             Normalize 0–255 → 0–1
                       │
                       ▼
             Random Forest Model
                       │
                       ▼
                 predict_proba()
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           Cat       Dog       Wild
             │         │         │
             └─────────┼─────────┘
                       ▼
                Display Results
```

---

# 💾 Model Serialization

The trained model was saved using Joblib.

```python
import joblib

joblib.dump(
    final_rf,
    "animal_classifier_compressed.pkl",
    compress=3
)
```

The compressed model is approximately:

```text
17 MB
```

The model is loaded by the Streamlit application:

```python
model = joblib.load(
    "animal_classifier_compressed.pkl"
)
```

The compression significantly reduced the model file size compared with the original serialized model.

---

# 📁 Project Structure

```text
animal-image-classifier/
│
├── app.py
│
├── animal_classifier_compressed.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 📄 File Description

## `app.py`

Contains the Streamlit application.

It handles:

* UI
* Image uploading
* Image preprocessing
* Model loading
* Prediction
* Probability calculation
* Result display

---

## `animal_classifier_compressed.pkl`

Contains the trained and compressed Random Forest model.

---

## `requirements.txt`

Contains the Python dependencies required by the application.

Current versions:

```text
streamlit==1.64.0
numpy==2.5.3
pillow==12.3.0
scikit-learn==1.9.1
joblib==1.6.0
```

---

## `README.md`

Contains the documentation for the project, including:

* Project description
* Dataset
* Methodology
* Experiments
* Results
* Installation
* Deployment
* Limitations
* Future improvements

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Data Processing

* NumPy
* Pillow

## Machine Learning

* Scikit-learn
* Random Forest Classifier
* Linear SVM

## Model Serialization

* Joblib

## Web Application

* Streamlit

## Version Control

* Git
* GitHub

## Deployment

* Streamlit Community Cloud

---

# 📦 Dependencies

The application uses the following package versions:

```text
streamlit==1.64.0
numpy==2.5.3
pillow==12.3.0
scikit-learn==1.9.1
joblib==1.6.0
```

Streamlit Community Cloud uses the repository's dependency file to construct the deployment environment. Keeping the dependency versions explicit helps make the deployment environment reproducible.

---

# 💻 Local Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/animal-image-classifier.git
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Navigate to the Project

```bash
cd animal-image-classifier
```

---

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

### Linux/macOS

```bash
python3 -m venv venv
```

---

## 4. Activate the Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Start the Application

```bash
streamlit run app.py
```

The application will normally become available at:

```text
http://localhost:8501
```

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

### Production Application

🚀 **https://animalimageclassifier.streamlit.app**

The repository contains the Streamlit entrypoint, trained model, and dependency file required for deployment.

Streamlit Community Cloud deploys an app by selecting the GitHub repository, branch, and entrypoint file. It then builds the environment from the declared dependencies.

---

## Deployment Configuration

```text
Repository:
animal-image-classifier

Branch:
main

Main file:
app.py
```

---

# 🔁 Deployment Workflow

```text
Developer
    │
    ▼
Modify Python Code
    │
    ▼
Commit Changes
    │
    ▼
Push to GitHub
    │
    ▼
GitHub Repository
    │
    ▼
Streamlit Community Cloud
    │
    ▼
Install Dependencies
    │
    ▼
Run app.py
    │
    ▼
Live Application
    │
    ▼
https://animalimageclassifier.streamlit.app
```

Changes pushed to the GitHub repository can trigger updates to the deployed Streamlit application. Streamlit Community Cloud also detects dependency changes and reinstalls the declared dependencies when necessary.

---

# ⚠️ Model Limitations

Although the final model achieves approximately 82.60% validation accuracy, it has several limitations.

---

## 1. Limited Number of Classes

The model only recognizes:

```text
Cat
Dog
Wild
```

It is not a general-purpose animal identification system.

---

## 2. Dataset Dependency

The model's performance depends on the characteristics of the training dataset.

Images significantly different from the training data may result in incorrect predictions.

---

## 3. Background Sensitivity

Since the model uses flattened raw pixel values, background information may influence the prediction.

---

## 4. Image Quality

Very blurry, dark, distorted, or heavily obstructed images may produce unreliable predictions.

---

## 5. Multiple Animals

The model produces one classification for the entire uploaded image.

For example, an image containing:

```text
Dog + Cat
```

will still receive one overall prediction.

The model does not detect and classify individual animals separately.

---

## 6. Classification vs Object Detection

This application performs **image classification**, not object detection.

It does not:

* Draw bounding boxes
* Locate individual animals
* Count animals
* Identify multiple objects separately

---

## 7. Confidence Is Not a Guarantee

The displayed confidence/probability should not be interpreted as a guarantee that the prediction is correct.

A Machine Learning model can be highly confident in an incorrect prediction, especially when an image is outside the training distribution.

---

# 🔮 Future Improvements

The project can be further improved in several ways.

---

## 🧠 Deep Learning

Replace the traditional Random Forest approach with a Convolutional Neural Network.

Potential architectures include:

* CNN
* ResNet
* EfficientNet
* MobileNet
* Vision Transformer

---

## 🔄 Transfer Learning

Use pretrained models such as:

```text
ResNet50
EfficientNet
MobileNetV2
```

and fine-tune them using the animal dataset.

This could provide more powerful visual feature extraction.

---

## 🖼️ Data Augmentation

Introduce techniques such as:

* Random rotation
* Horizontal flipping
* Random cropping
* Zooming
* Brightness adjustment
* Contrast adjustment

This can help the model generalize better to unseen images.

---

## 🐾 More Animal Classes

The application could be expanded to include additional categories such as:

```text
Horse
Elephant
Tiger
Lion
Bear
Bird
Rabbit
Cow
```

---

## 🎯 Object Detection

Object detection models such as:

* YOLO
* Faster R-CNN
* SSD

could be used to detect multiple animals within the same image.

---

## 🔍 Explainable AI

Future versions could include explainability techniques such as:

* SHAP
* Feature importance
* Grad-CAM

to provide more insight into why a model produced a prediction.

---

## 📊 Additional Application Features

Possible future features include:

* Prediction history
* Batch image classification
* Downloadable prediction reports
* Confusion matrix visualization
* Model comparison dashboard
* Image preprocessing controls
* Multi-animal detection
* User feedback on predictions

---

# 📚 Machine Learning Concepts Demonstrated

This project demonstrates practical experience with:

### Data Processing

* Image loading
* Image resizing
* RGB conversion
* NumPy arrays
* Feature transformation
* Pixel normalization

### Machine Learning

* Supervised learning
* Multi-class classification
* Random Forest
* Linear SVM
* Hyperparameter tuning
* Model comparison

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Class-wise evaluation

### Deployment

* Model serialization
* Joblib
* Streamlit
* Git
* GitHub
* Streamlit Community Cloud

---

# 🎓 Learning Outcomes

This project provided practical experience in building a Machine Learning application from beginning to end.

### Python

Experience with:

```text
NumPy
Pillow
Scikit-learn
Joblib
Streamlit
```

### Machine Learning

Experience with:

```text
Image preprocessing
Feature engineering
Classification
Random Forest
SVM
Hyperparameter tuning
Model evaluation
```

### Deployment

Experience with:

```text
Model serialization
Streamlit application development
GitHub repository management
Cloud deployment
Dependency management
```

---

# 📌 Key Project Statistics

| Property              | Value                     |
| --------------------- | ------------------------- |
| Dataset               | AFHQ                      |
| Classes               | 3                         |
| Training Images       | 14,630                    |
| Validation Images     | 1,500                     |
| Total Images          | 16,130                    |
| Input Resolution      | 64 × 64                   |
| Color Format          | RGB                       |
| Features per Image    | 12,288                    |
| Final Algorithm       | Random Forest             |
| Number of Trees       | 300                       |
| Validation Accuracy   | **82.60%**                |
| Model Format          | Joblib                    |
| Compressed Model Size | ~17 MB                    |
| Web Framework         | Streamlit                 |
| Deployment            | Streamlit Community Cloud |

---

# 🏁 Conclusion

**AI Animal Vision** demonstrates a complete Machine Learning workflow for image classification.

The project starts with an image dataset and progresses through:

```text
Data
 ↓
Preprocessing
 ↓
Feature Extraction
 ↓
Model Training
 ↓
Hyperparameter Tuning
 ↓
Evaluation
 ↓
Model Serialization
 ↓
Web Application
 ↓
Cloud Deployment
```

The final Random Forest model achieved approximately **82.60% validation accuracy** on the three selected animal categories.

The model was then integrated into a Streamlit application that provides image uploading, prediction, confidence scores, class probabilities, and a professional user interface.

### 🚀 Try the live application:

**https://animalimageclassifier.streamlit.app**

---

# 👨‍💻 Author

## Tej

**Machine Learning | Python | Data Science**

This project was developed as a practical Machine Learning and deployment project.

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

# 📜 License

This project is intended for educational, learning, and portfolio purposes.

The original dataset may have its own licensing and usage requirements. Please refer to the dataset's original source and license before redistributing the dataset or using it for commercial purposes.

