"""
CNN model for cats vs dogs classification.
Built from scratch — no pretrained weights.
"""

import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    """
    A 4-block CNN for binary image classification.
    
    Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
    Classifier: Flatten -> FC -> ReLU -> Dropout -> FC -> Sigmoid
    
    Expects 128x128 RGB input images.
    """
    
    def __init__(self, dropout_rate=0.5):
        super(CatDogCNN, self).__init__()
        
        # Block 1: 3 channels -> 32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        )
        
        # Block 2: 32 -> 64 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        
        # Block 3: 64 -> 128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )
        
        # Block 4: 128 -> 256 filters
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)  # single output for binary classification
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Quick helper to count trainable params."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


if __name__ == "__main__":
    # Sanity check — make sure the shapes work out
    model = CatDogCNN()
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
