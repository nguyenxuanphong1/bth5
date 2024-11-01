import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# Function to load dental images
def load_dental_images(folder_path, image_size=(64, 64)):
    images = []
    labels = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                # Tạo nhãn tự động hoặc điều chỉnh theo quy tắc của bạn
                label = idx  # Dùng số thứ tự làm nhãn
                labels.append(label)
    images = np.array(images).reshape(len(images), -1)  # Flatten images
    return images, np.array(labels)


# Load IRIS dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Tải tập dữ liệu ảnh (sử dụng đường dẫn tuyệt đối)
folder_path = os.path.abspath('D:/XLATHGIMAYTINH/bth5/anhtest/test1')
X_dental, y_dental = load_dental_images(folder_path)


# Define classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "CART (Gini)": DecisionTreeClassifier(criterion='gini'),
    "ID3 (Info Gain)": DecisionTreeClassifier(criterion='entropy'),
    "Neuron": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
}

# Function to train and evaluate classifiers on a given dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Sử dụng zero_division=1 trong hàm evaluate_classifiers()
def evaluate_classifiers(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro', zero_division=1),
            "Recall": recall_score(y_test, y_pred, average='macro', zero_division=1)
        }
    return results

# Evaluate on IRIS dataset
print("Results for IRIS dataset:")
iris_results = evaluate_classifiers(X_iris, y_iris)
for name, metrics in iris_results.items():
    print(f"{name} - Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, Recall: {metrics['Recall']}")

# Evaluate on Dental Images dataset
print("\nResults for Dental Images dataset:")
dental_results = evaluate_classifiers(X_dental, y_dental)
for name, metrics in dental_results.items():
    print(f"{name} - Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, Recall: {metrics['Recall']}")
print("Unique labels in dental dataset:", np.unique(y_dental))