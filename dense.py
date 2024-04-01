import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load the data
data = np.load('emnist_letters.npz')
train_images = data['train_images']
train_labels = data['train_labels']
validate_images = data['validate_images']
validate_labels = data['validate_labels']
test_images = data['test_images']
test_labels = data['test_labels']

# Convert to PyTorch tensors and ensure labels are long type
train_images = torch.from_numpy(train_images).float()
train_labels = torch.from_numpy(train_labels).long()
validate_images = torch.from_numpy(validate_images).float()
validate_labels = torch.from_numpy(validate_labels).long()
test_images = torch.from_numpy(test_images).float()
test_labels = torch.from_numpy(test_labels).long()

# Adjusting the labels if they are in a format other than 1D tensor of class indices
train_labels = torch.argmax(train_labels, axis=1) if train_labels.dim() > 1 else train_labels
validate_labels = torch.argmax(validate_labels, axis=1) if validate_labels.dim() > 1 else validate_labels
test_labels = torch.argmax(test_labels, axis=1) if test_labels.dim() > 1 else test_labels

# Adjust the labels to be 0-based if they are 1-based
train_labels = train_labels - 1
validate_labels = validate_labels - 1
test_labels = test_labels - 1

# Create TensorDataset objects
train_dataset = TensorDataset(train_images, train_labels)
validate_dataset = TensorDataset(validate_images, validate_labels)
test_dataset = TensorDataset(test_images, test_labels)

# Create DataLoader objects
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten the image input
        #print(f"Shape after flatten: {x.shape}")
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        #print(f"Output shape: {x.shape}")
        return x

# Assuming each image in EMNIST Letters is 28x28 pixels
input_size = 28 * 28 
hidden_sizes = [512, 256, 128]
output_size = 26  # For 26 letters

# Create the network
model = DenseNet(input_size, hidden_sizes, output_size)

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training
def train(model, train_loader, loss_fn, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            #print(f"Batch {batch} - X shape: {X.shape}, y shape: {y.shape}")
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = evaluate(model, train_loader)
        validate_accuracy = evaluate(model, validate_loader)
        print(f'Epoch {epoch}: Train Accuracy: {train_accuracy:.4f}, Validate Accuracy: {validate_accuracy:.4f}')

# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            predicted = torch.argmax(pred, axis=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# Training the model
train(model, train_loader, loss_fn, optimizer)

# Evaluating the model
accuracy = evaluate(model, test_loader)
print(f'Test accuracy: {accuracy}')
