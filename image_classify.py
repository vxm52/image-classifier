import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
        logits = self.linear_relu_stack(x)
        return logits


# Instantiate the model to a GPU
model = ImageClassifier().to("cuda")

X = "data placeholder"

logits = model(X)  # Pass the model some data (this is automatically call the forward method for training and prediction).

pred_probability = nn.Softmax(dim=1)(logits)
y_pred = pred_probability.argmax(1)

print(f"And my predictions is... {y_pred}")