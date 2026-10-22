"""
PyTorch Convolutional Neural Network (CNN) for Image Classification.
Implements a modular ConvNet architecture with convolutional blocks, pooling,
batch normalization, dropout regularization, and sample inference check.
"""

import sys

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ImageClassifierCNN(nn.Module):
    """
    Standard 2D Convolutional Neural Network.
    Accepts images of shape (B, in_channels, H, W).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves spatial dims

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves spatial dims

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fix output feature map size to 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


def test_cnn_architecture():
    if not TORCH_AVAILABLE:
        print("[Notice] PyTorch is not installed in the active environment.")
        print("Please install PyTorch using: pip install torch")
        return

    print("Testing ImageClassifierCNN with dummy input tensor...")
    model = ImageClassifierCNN(in_channels=1, num_classes=10)
    print(model)

    # Simulate batch of 8 grayscale images (e.g., MNIST/FashionMNIST 28x28)
    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)

    print(f"\nInput Tensor Shape:  {dummy_input.shape}")
    print(f"Output Logits Shape: {output.shape}")
    print("Forward pass successful! Model is ready for dataset integration.")


if __name__ == "__main__":
    test_cnn_architecture()
