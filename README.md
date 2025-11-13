CNN-Based Breast Cancer Image Classification: Malignant vs Benign
Objective

Develop and evaluate a Convolutional Neural Network (CNN) model to classify breast cancer histopathology images into malignant and benign categories. The goal is to enhance early detection and diagnostic accuracy, providing a faster and more reliable alternative to traditional diagnostic methods and manual examination.

Problem Statement

Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Early and accurate diagnosis is critical for effective treatment and improved survival rates.

Traditional diagnostic methods, such as manual examination of histopathology slides, are:

Time-consuming

Prone to human error

Highly dependent on pathologist expertise

Existing machine learning approaches often rely on handcrafted features, which may not capture the complex patterns in histopathology images.

Thus, there is a need for an automated, reliable, and accurate system that can classify breast cancer images into malignant and benign categories, enabling faster and more precise diagnosis.

Data Description

Dataset Name: Breast Histopathology Images

Source: Kaggle

Overview

The dataset contains histopathology images of breast tissue extracted from whole-slide images of breast cancer biopsies.

Labels: Malignant or Benign

Image Type: RGB (colored)

Level: Cellular-level tissue representation

The dataset is suitable for training CNNs due to its detailed representation of tissue structures, which allows the network to learn discriminative features automatically.

Model Approach

Data Preprocessing

Resizing images to a consistent input shape

Normalization of pixel values

Optional: Data augmentation to improve generalization (rotation, flipping, zooming)

CNN Architecture

Multiple convolutional layers with ReLU activations

Pooling layers for spatial downsampling

Dropout layers for regularization

Fully connected layers for classification

Training

Optimizer: Adam

Loss function: Categorical Cross-Entropy

Class weighting applied to handle mild class imbalance

Evaluation

Metrics: Accuracy, Precision, Recall, F1-score

Confusion matrix to visualize class-level performance

Early stopping and learning rate reduction for optimized training

Results

Test Accuracy: 87.37%

Classification Performance:

Benign: High recall (0.95), F1-score = 0.89

Malignant: High precision (0.92), F1-score = 0.85

The model shows good generalization with minimal overfitting.

Conclusion

The CNN-based model effectively distinguishes between malignant and benign breast cancer images. It demonstrates:

Reliable performance on unseen test data

Strong recall for benign cases

High precision for malignant cases

The system provides a robust automated tool to aid pathologists, potentially improving early diagnosis and treatment decisions.
