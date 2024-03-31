import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load the data
data = np.load('emnist_letters.npz')
train_images = data['train_images']
train_labels = data['train_labels']
validate_images = data['validate_images']
validate_labels = data['validate_labels']
test_images = data['test_images']
test_labels = data['test_labels']

# Normalize the images
train_images = train_images.astype('float32') / 255
validate_images = validate_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Convert to PyTorch tensors
train_images = torch.from_numpy(train_images)
train_labels = torch.from_numpy(train_labels)
validate_images = torch.from_numpy(validate_images)
validate_labels = torch.from_numpy(validate_labels)
test_images = torch.from_numpy(test_images)
test_labels = torch.from_numpy(test_labels)

# Create TensorDataset objects
train_dataset = TensorDataset(train_images, train_labels)
validate_dataset = TensorDataset(validate_images, validate_labels)
test_dataset = TensorDataset(test_images, test_labels)

# Create DataLoader objects
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
