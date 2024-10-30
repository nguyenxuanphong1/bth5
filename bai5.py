import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# Function to load and augment dental images
def load_dental_images(folder_path, image_size=(64, 64), num_classes=10):
    images = []
    labels = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                label = idx % num_classes  # Reduce labels to a specified number of classes
                labels.append(label)
    images = np.array(images).reshape(len(images), -1)  # Flatten images
    return images, np.array(labels)

# Data augmentation function
def augment_image(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    flipped = cv2.flip(img, 1)
    return [img, rotated, flipped]

# Load and augment dental images
def load_and_augment_images(folder_path, image_size=(64, 64), num_classes=10):
    images, labels = load_dental_images(folder_path, image_size, num_classes)
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        aug_imgs = augment_image(img.reshape(image_size))  # Reshape to image size for augmentation
        augmented_images.extend([i.flatten() for i in aug_imgs])  # Flatten augmented images
        augmented_labels.extend([label] * len(aug_imgs))
    return np.array(augmented_images), np.array(augmented_labels)

# Load IRIS dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Load and augment dental images dataset
X_dental, y_dental = load_and_augment_images('anhtest/Patch1', image_size=(64, 64), num_classes=10)

# Define classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "CART (Gini)": DecisionTreeClassifier(criterion='gini'),
    "ID3 (Info Gain)": DecisionTreeClassifier(criterion='entropy'),
    "Neuron": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
}

# Function to train and evaluate classifiers on a given dataset
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
print("\nResults for Processed Dental Images dataset:")
dental_results = evaluate_classifiers(X_dental, y_dental)
for name, metrics in dental_results.items():
    print(f"{name} - Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, Recall: {metrics['Recall']}")
print("Unique labels in dental dataset:", np.unique(y_dental))
