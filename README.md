# Simple CNN for Image Classification using PyTorch

This script implements a simple Convolutional Neural Network (CNN) using PyTorch for image classification. The CNN consists of two convolutional layers followed by max-pooling and fully connected layers.

## Model Architecture
![diagram](https://github.com/Marinto-Richee/Photo-Classifier-Pytorch/assets/65499285/8d34c956-286f-4378-aca8-20b426fa61b1)


The CNN architecture is defined using the `SimpleCNN` class, which inherits from `nn.Module`:

```plaintext
SimpleCNN(
  (conv1): Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (conv2): Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
  (relu2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (fc1): Linear(in_features=65536, out_features=128, bias=True)
  (relu3): ReLU()
  (fc2): Linear(in_features=128, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```

## Data Preprocessing and Loading

### Setting the Random Seed for Reproducibility

To ensure reproducibility of the results, the random seed is set using the following code:

```python
torch.manual_seed(42)
```

This sets the random seed for PyTorch operations.

### Data Transforms

The dataset is preprocessed using the following data transformations:

```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
```

The transforms include resizing the images to 128x128 pixels and converting them to tensors.

### Loading the Dataset

The dataset is loaded using the `ImageFolder` class from torchvision:

```python
dataset = datasets.ImageFolder(root="iamges", transform=transform)
```

The dataset is structured in a folder format, and each subfolder represents a class.

### Dataset Split

The dataset is split into training, validation, and test sets using `random_split`:

```python
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
```

80% of the data is used for training, 10% for validation, and the remaining 10% for testing.

### Data Loaders

Data loaders are created to efficiently load the data in batches during training and evaluation:

```python
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

The `batch_size` is set to 32, and the training data loader shuffles the data for each epoch to improve model learning.

## Loss Function and Optimizer

The loss function used for this binary classification task is the Binary Cross-Entropy Loss (BCELoss). The optimizer employed is Adam with a learning rate of 0.001.

```python
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Training Loop

The model is trained for 10 epochs using the training data, with the BCELoss as the optimization criterion and Adam as the optimizer.

```python
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float().unsqueeze(1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    # ... (validation code)

    # Printing epoch-wise training details
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

## Testing Loop

The model is evaluated on the test set to calculate the test loss and accuracy.

```python
# ... (testing code)

test_accuracy = correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
```

The test loss and accuracy are printed to evaluate the model's performance on the unseen test data.

## Saving the Model

The model's state dictionary is saved to a file for future use.

```python
torch.save(model.state_dict(), 'Photo_classifierV1.pth')
```

This Markdown provides an explanation of the model architecture, data preprocessing and loading, loss function, optimizer, training loop, validation, testing, and model saving in the provided PyTorch code.

## Usage

To use the saved model for making predictions or further training, load the model as follows:

```python
# Load the model
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('Photo_classifierV1.pth'))
loaded_model.eval()
```

Now, `loaded_model` is ready for making predictions or further training.

## Dependencies

- PyTorch
- torchvision
- PIL (Pillow)

Ensure you have the required dependencies installed to run the code successfully.

## License

This code is licensed under the [MIT License](LICENSE).
