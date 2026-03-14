"""
Dataset class and image transforms for cats vs dogs.
Handles loading images from folders and applying augmentations.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# --- Image Transforms ---

def get_train_transforms(img_size=128):
    """Augmented transforms for training — helps reduce overfitting."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size=128):
    """Clean transforms for validation/test — no augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# --- Dataset ---

class CatDogDataset(Dataset):
    """
    Custom dataset for cats vs dogs images.
    
    Expects a list of (image_path, label) tuples.
    Label: 0 = cat, 1 = dog
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB (some might be grayscale)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_image_paths(data_dir):
    """
    Load all image paths and labels from a directory.
    
    Assumes filenames like 'cat.0.jpg', 'dog.0.jpg' (Kaggle format)
    or subfolders 'cats/' and 'dogs/'.
    
    Returns:
        image_paths: list of file paths
        labels: list of ints (0=cat, 1=dog)
    """
    image_paths = []
    labels = []
    
    # Check if data is in subfolders (cats/, dogs/)
    cats_dir = os.path.join(data_dir, "cats")
    dogs_dir = os.path.join(data_dir, "dogs")
    
    if os.path.isdir(cats_dir) and os.path.isdir(dogs_dir):
        # Subfolder structure
        for fname in os.listdir(cats_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cats_dir, fname))
                labels.append(0)
        
        for fname in os.listdir(dogs_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(dogs_dir, fname))
                labels.append(1)
    else:
        # Flat structure — filenames start with 'cat' or 'dog'
        for fname in os.listdir(data_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            fpath = os.path.join(data_dir, fname)
            if fname.lower().startswith("cat"):
                image_paths.append(fpath)
                labels.append(0)
            elif fname.lower().startswith("dog"):
                image_paths.append(fpath)
                labels.append(1)
    
    print(f"Loaded {len(image_paths)} images from {data_dir}")
    print(f"  Cats: {labels.count(0)}, Dogs: {labels.count(1)}")
    
    return image_paths, labels
