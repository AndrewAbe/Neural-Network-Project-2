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
def train(model, train_loader, loss_fn, optimizer, num_epochs=0): #######################CHANGE
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





#CONVOLUTIONAL
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Assuming input images are 28x28 pixels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1 input channel, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After two pooling operations, 28x28 becomes 7x7
        self.fc2 = nn.Linear(128, 26)  # Assuming 26 output classes for letters

    def forward(self, x):
        # Reshape the input to have the channels dimension
        x = x.view(-1, 1, 28, 28)  # Assuming x is of shape [batch_size, 784]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create the network
model = ConvNet()

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train(model, train_loader, loss_fn, optimizer, num_epochs=0): #############################CHANGE
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            # Forward pass: compute the predicted outputs by passing inputs to the model
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #print(f'Epoch {epoch}: Loss: {total_loss/len(train_loader)}')

        # Evaluate after each epoch
        train_accuracy = evaluate(model, train_loader)
        validate_accuracy = evaluate(model, validate_loader)
        print(f'Epoch {epoch}: Train Accuracy: {train_accuracy:.4f}, Validate Accuracy: {validate_accuracy:.4f}')

def evaluate(model, loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed
        for X, y in loader:
            pred = model(X)
            predicted = torch.argmax(pred, axis=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy

# Assuming you have the DataLoader and CNN model set up
model = ConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
train(model, train_loader, loss_fn, optimizer)

# Evaluate the model
test_accuracy = evaluate(model, test_loader)
print(f'Test accuracy: {test_accuracy:.4f}')




#Generative adversarial network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: Latent vector z (size of 100)
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Size: (512, 4, 4)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Size: (256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Size: (128, 16, 16)
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output Size: (1, 28, 28)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: Image (1, 28, 28)
            nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (128, 14, 14)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (256, 7, 7)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (512, 4, 4)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: a single scalar
        )

    def forward(self, img):
        return self.model(img)

# Create instances of the Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

import torch

# Number of epochs to train for
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):  # Assuming images are the input data
        # Training Discriminator
        # Generate batch of latent vectors
        noise = torch.randn(images.size(0), 100, 1, 1, device=device)

        # Generate fake image batch with G
        fake_images = generator(noise)

        # Classify all fake batch with D
        discriminator_real = discriminator(images).view(-1)
        loss_D_real = criterion(discriminator_real, torch.ones_like(discriminator_real))

        discriminator_fake = discriminator(fake_images.detach()).view(-1)
        loss_D_fake = criterion(discriminator_fake, torch.zeros_like(discriminator_fake))

        # Accumulate discriminator loss and backpropagate
        loss_D = (loss_D_real + loss_D_fake) / 2
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Training Generator
        # Classify all fake batch with D
        output = discriminator(fake_images).view(-1)
        loss_G = criterion(output, torch.ones_like(output))

        # Backpropagate generator loss
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    # Output training stats
    if (i+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}')