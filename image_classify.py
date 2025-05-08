import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the neural network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # Flatten layer takes a multi-dimensional input like an image and convert it into 1-dimension.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # Fully connected layer that transforms the flattened 28x28 image into an output of 512.
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # Outputs the 10 labels the model is trying to predict.
        )

    # Define the flow of data
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Define transformations and load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Instantiate the model to a GPU
model = ImageClassifier().to("cuda")

X = "data placeholder"

logits = model(X)  # Pass the model some data (this is automatically call the forward method for training and prediction).

pred_probability = nn.Softmax(dim=1)(logits)
y_pred = pred_probability.argmax(1)

print(f"And my predictions is... {y_pred}")