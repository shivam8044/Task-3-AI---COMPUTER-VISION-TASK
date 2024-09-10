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

