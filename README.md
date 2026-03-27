# Sports Ball Classification - Multi-Class Image Classification

## 📌 Project Overview
This project implements a **multi-class image classification model** to identify different types of sports balls from images. The dataset contains 15 categories of sports balls and related equipment, with over 7,000 images.

### 🏀 Categories Classified
- American Football
- Baseball
- Basketball
- Billiard Ball
- Bowling Ball
- Cricket Ball
- Football (Soccer)
- Golf Ball
- Hockey Ball
- Hockey Puck
- Rugby Ball
- Shuttlecock (Badminton)
- Table Tennis Ball
- Tennis Ball
- Volleyball

## 📊 Dataset Information
- **Source**: [Kaggle - Sports Balls Multi-class Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification)
- **Total Images**: 7,328
- **Training Split**: 80% (5,863 images)
- **Validation Split**: 20% (1,465 images)
- **Image Size**: 224x224 pixels
- **License**: CC0-1.0

## 🏗️ Model Architecture

### 1. Custom ResNet-50 Style Model
A ResNet-50 style architecture built from scratch with residual blocks:

Initial Conv2D (64 filters, 7x7, stride 2)

Batch Normalization + MaxPooling

Stage 1: 3 residual blocks (64 filters)

Stage 2: 4 residual blocks (128 filters) with stride 2

Stage 3: 6 residual blocks (256 filters) with stride 2

Stage 4: 3 residual blocks (512 filters) with stride 2

Global Average Pooling

Dense Layer (15 units, softmax)


**Total Parameters**: ~23.6M

### 2. Transfer Learning with ResNet50
Pre-trained ResNet50 (ImageNet weights) with custom classification head:

ResNet50 base (frozen)

Global Average Pooling

Dense Layer (256 units, ReLU)

Dropout (0.5)

Dense Layer (15 units, softmax)


**Total Parameters**: ~24.1M (528K trainable)

## 🔧 Requirements

```python
tensorflow>=2.0
numpy
matplotlib
seaborn
pandas
kaggle

📈 Training Results
Custom ResNet-50 Style Model (5 epochs)
Epoch	Training Accuracy	Training Loss	Validation Accuracy	Validation Loss
1	16.9%	2.88	8.5%	6.72
2	24.4%	2.43	20.9%	2.48
3	26.9%	2.32	16.6%	2.60
4	30.7%	2.19	30.4%	2.30
5	33.6%	2.12	26.4%	2.38
Transfer Learning with ResNet50 (5 epochs)
Epoch	Training Accuracy	Training Loss	Validation Accuracy	Validation Loss
1	10.8%	2.70	12.5%	2.62
2	12.3%	2.62	13.5%	2.59
3	13.0%	2.60	16.9%	2.57
4	14.3%	2.59	15.7%	2.57
5	15.2%	2.56	18.8%	2.54
🔍 Data Visualization
The notebook includes:

Class Distribution Bar Chart: Shows image count per category

Random Image Grid: Displays 9 random images from the dataset with their labels

Training History Plots: Accuracy and loss curves over epochs

## 🎯 Future Improvements
Fine-tuning: Unfreeze later layers of ResNet50 and train with lower learning rate

More Epochs: Train for 20-50 epochs for better convergence

Learning Rate Scheduling: Implement cosine decay or ReduceLROnPlateau

Hyperparameter Tuning: Optimize batch size, dropout rate, and learning rate

Model Ensembling: Combine multiple models for better accuracy

Deployment: Create a web app using Flask/Streamlit for real-time predictions
