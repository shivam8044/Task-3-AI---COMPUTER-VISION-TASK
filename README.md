# Task-3-AI---COMPUTER-VISION-TASK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load the Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Neural Network model
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, activation='relu', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Extracting metrics for the bar graph
metrics_df = report_df[['precision', 'recall', 'f1-score']].iloc[:-1]  # Exclude 'accuracy' row

# Plotting the metrics as a bar graph
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Classification Metrics by Class')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Metrics')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load the Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Neural Network model
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, activation='relu', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Extracting metrics for the bar graph
metrics_df = report_df[['precision', 'recall', 'f1-score']].iloc[:-1]  # Exclude 'accuracy' row

# Plotting the metrics as a bar graph
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Classification Metrics by Class')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Metrics')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Bar plot for classification report
plt.figure(figsize=(10, 6))
report_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar')
plt.title('Classification Report')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load the Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Neural Network model
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, activation='relu', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Extracting metrics for the bar graph (excluding 'accuracy' row)
metrics_df = report_df[['precision', 'recall', 'f1-score']].iloc[:-1]

# Plotting the metrics as a bar graph
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Classification Metrics by Class')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Metrics')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Evaluate the model
y_pred = model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Display the classification report as a DataFrame
print(report_df)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Step 1: Load the Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train a Neural Network model
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, activation='relu', random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Step 6: Generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Step 7: Extracting metrics for the bar graph
metrics_df = report_df[['precision', 'recall', 'f1-score']].iloc[:-1]  # Exclude 'accuracy' row

# Step 8: Plotting the metrics as a bar graph
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Classification Metrics by Class')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.ylim(0, 1)  # Set the y-axis range from 0 to 1
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Metrics')
plt.grid(True, linestyle='--', alpha=5)
plt.tight_layout()
plt.show()


Name :SHIVAM RAJ 
company : CODE TECH IT SOLUTION 
ID: CT08DS7115 
DOMAIN: Artificial Intelligence 
DURATION : August To September2024 MENTOR :SRAVANI GOUNI

#Overview of COMPUTER VISION a Project :-

Computer Vision (CV) is a field within artificial intelligence (AI) that enables computers to interpret and understand the visual world. By using digital images from cameras, videos, and deep learning models, machines can accurately identify and classify objects, recognize patterns, and perform various tasks that require visual input.

Here’s an overview of the key stages and components of a typical computer vision project:

1. Project Definition and Objectives
Problem Statement: Clearly define the problem you are trying to solve using computer vision (e.g., object detection, image classification, facial recognition, image segmentation).
Scope and Goals: Establish the desired outcome (e.g., detect objects with 95% accuracy, identify people in real-time).
2. Data Collection
Source Identification: Gather images, videos, or sensor data required for the project.
Public datasets (ImageNet, COCO, PASCAL VOC, etc.).
Custom data collection using cameras, drones, or specialized equipment.
Data Types: Image (2D or 3D), video streams, medical scans (e.g., MRI), etc.
Data Labeling: Annotate the data with the necessary labels (bounding boxes, segmentation masks, or class labels).
Manual Labeling: Human annotators provide labels.
Automated Labeling: Pre-trained models or tools can assist in generating labels.
3. Data Preprocessing
Normalization: Scale pixel values (e.g., from 0 to 255 to 0 to 1).
Resizing: Resize images to a consistent size suitable for the model (e.g., 224x224 for ResNet).
Augmentation: Apply transformations (rotation, flipping, cropping, zooming) to increase the diversity of the training data and prevent overfitting.
Noise Removal: Techniques like blurring or denoising are used to improve image quality.
4. Model Selection
Architecture Choice: Depending on the task, select an appropriate model architecture, such as:
Image Classification: Convolutional Neural Networks (CNNs) like ResNet, VGG, Inception.
Object Detection: YOLO (You Only Look Once), Faster R-CNN, SSD (Single Shot Multibox Detector).
Segmentation: UNet, Mask R-CNN, Fully Convolutional Networks (FCNs).
Other tasks: GANs for image generation, Autoencoders for anomaly detection.
5. Model Training
Split the Dataset: Typically into training, validation, and test sets (e.g., 70% for training, 15% for validation, 15% for testing).
Loss Function: Choose an appropriate loss function (e.g., cross-entropy for classification, IoU for segmentation).
Optimization: Apply an optimizer such as Adam, SGD, or RMSProp to minimize the loss function.
Hyperparameters: Tuning learning rates, batch sizes, and epochs is crucial for effective training.
Transfer Learning: Use pre-trained models to speed up training when working with limited data.
6. Model Evaluation
Metrics: Common evaluation metrics for computer vision projects include:
Accuracy: For classification problems.
Precision, Recall, F1-Score: For detection tasks.
IoU (Intersection over Union): For object detection and segmentation tasks.
Confusion Matrix: To visualize model performance in classification tasks.
Cross-validation: Helps in ensuring the model’s generalizability by testing on different subsets of data.
7. Model Optimization
Hyperparameter Tuning: Adjust learning rates, batch sizes, and epochs to optimize performance.
Regularization: Techniques like dropout or weight decay help avoid overfitting.
Model Pruning: Reducing the complexity of the model without sacrificing much accuracy.
Quantization: Reducing the precision of the model weights for deployment on lower-power devices (e.g., smartphones).
8. Deployment
Model Conversion: Convert models into formats suitable for deployment (e.g., TensorFlow Lite, ONNX, CoreML).
Edge Deployment: Deploying models on edge devices such as smartphones, cameras, or embedded systems.
Cloud Deployment: Hosting models on cloud services (e.g., AWS, Azure, Google Cloud).
APIs: Create APIs for interacting with the model in real-time.
9. Post-Deployment Monitoring
Performance Monitoring: Ensure the model continues to perform well in real-world settings.
Retraining: Continuously update and improve the model based on new data.
Error Analysis: Identifying areas where the model fails and iterating on improvements.
Common Applications of Computer Vision Projects:
Autonomous Driving: Object detection, lane tracking, pedestrian detection.
Medical Imaging: Detecting tumors, analyzing MRI or X-ray scans.
Facial Recognition: Identifying individuals in security systems or for authentication.
Retail: Image-based product search, checkout-free stores.

![Screenshot 2024-09-10 192323](https://github.com/user-attachments/assets/54ca61df-4ef9-45ee-a6ab-2b5fd46ce63d)
![Screenshot 2024-09-10 192349](https://github.com/user-attachments/assets/52ed9506-45a1-4673-a8a8-3acd1a76ddff)

