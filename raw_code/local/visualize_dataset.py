import os
import tarfile
from PIL import Image
import matplotlib.pyplot as plt
import random

# Specify the path to the downloaded .tgz file

def split_classes(labels, train_ratio=0.8):
    """Splits class labels into training and testing sets."""
    unique_classes = list(set(labels))
    random.shuffle(unique_classes)
    num_train = int(len(unique_classes) * train_ratio)
    train_classes = unique_classes[:num_train]
    test_classes = unique_classes[num_train:]
    return train_classes, test_classes

def sample_task(images, labels, chosen_classes, k_support, k_query):
    """Samples a single few-shot learning task."""
    support_set = []
    query_set = []
    for c in chosen_classes:
        class_indices = [i for i, lbl in enumerate(labels) if lbl == c]
        chosen_indices = random.sample(class_indices, k_support + k_query)
        support_indices = chosen_indices[:k_support]
        query_indices = chosen_indices[k_support:]
        
        support_set.extend([(images[i], labels[i]) for i in support_indices])
        query_set.extend([(images[i], labels[i]) for i in query_indices])
    
    return support_set, query_set

# Set the path to the extracted dataset directory
dataset_dir = "CUB_200_2011"
# Paths to relevant files
images_file = os.path.join(dataset_dir, "images.txt")
labels_file = os.path.join(dataset_dir, "image_class_labels.txt")
image_dir = os.path.join(dataset_dir, "images")

# Read image filenames and their corresponding class labels
images = []
labels = []
with open(images_file, "r") as f:
    for line in f:
        _, img_filename = line.strip().split()
        images.append(img_filename)

with open(labels_file, "r") as f:
    for line in f:
        _, label = line.strip().split()
        labels.append(int(label))

# Verify the number of images matches the number of labels
assert len(images) == len(labels), "Mismatch between number of images and labels"


# Step 1: Split dataset into training/testing classes
train_classes, test_classes = split_classes(labels)

# Step 2: Create tasks
N = 5  # N-way classification
K_support = 5  # 1-shot
K_query = 15  # 5 queries per class

# Training tasks
train_tasks = []
for _ in range(100):  # Number of training tasks
    chosen_classes = random.sample(train_classes, N)
    support_set, query_set = sample_task(images, labels, chosen_classes, K_support, K_query)
    train_tasks.append((support_set, query_set))

# Testing tasks
test_tasks = []
for _ in range(20):  # Number of testing tasks
    chosen_classes = random.sample(test_classes, N)
    support_set, query_set = sample_task(images, labels, chosen_classes, K_support, K_query)
    test_tasks.append((support_set, query_set))


# Function to display a grid of images
def plot_images(images, title, rows=1):
    """Plots a grid of images."""
    plt.figure(figsize=(15, 5))
    for i, (img_path, label) in enumerate(images):
        img_path = os.path.join(image_dir, img_path)
        img = Image.open(img_path)
        plt.subplot(rows, len(images), i + 1)
        plt.imshow(img)
        plt.title(f"Class {label}")
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.show()

# Plotting the support set
plot_images(support_set, "Support Set")

# Plotting the query set
plot_images(query_set, "Query Set")