import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


class FlowerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def load_flower_dataset(data_dir, test_size=0.2, val_size=0.1, random_state=42):
    train_dir = os.path.join(data_dir, 'train')
    classes = sorted(os.listdir(train_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    image_paths = []
    labels = []

    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_to_idx[class_name])

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx


def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=4):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx = load_flower_dataset(data_dir)

    train_transform, val_transform = get_data_transforms(image_size)

    train_dataset = FlowerDataset(X_train, y_train, transform=train_transform)
    val_dataset = FlowerDataset(X_val, y_val, transform=val_transform)
    test_dataset = FlowerDataset(X_test, y_test, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_to_idx


if __name__ == '__main__':
    data_dir = 'data/flowers'
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(data_dir, batch_size=16)
    print(f"Classes: {class_to_idx}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
