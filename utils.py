import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import random
import torchvision

class ImageSegmentationDatasetOneHotEncoding(Dataset):
    def __init__(self, root_dir, transform=None, augment=True):
        self.root_dir = root_dir
        # because of .DS_Store presence
        self.folders = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        self.transform = transform
        self.augment = augment
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root_dir, self.folders[idx])

        rgb_path = os.path.join(folder_path, 'rgb.jpg')
        label_path = os.path.join(folder_path, 'labels.png')

        rgb_image = Image.open(rgb_path).convert('RGB')
        label_image = Image.open(label_path).convert('RGB')


        # augmentation
        if self.augment and random.random() < 0.5:
            rgb_image = TF.hflip(rgb_image)
            label_image = TF.hflip(label_image)

        label_onehotencoding = convert_rgb_to_onehotencoding(label_image)
        rgb_image = self.to_tensor(rgb_image)
        label_image = self.to_tensor(label_image)
        
        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, label_image, label_onehotencoding

class ImageSegmentationDatasetLogit(Dataset):
    def __init__(self, root_dir, transform=None, augment=True):
        self.root_dir = root_dir
        # because of .DS_Store presence
        self.folders = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        self.transform = transform
        self.augment = augment
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root_dir, self.folders[idx])

        rgb_path = os.path.join(folder_path, 'rgb.jpg')
        label_path = os.path.join(folder_path, 'labels.png')

        rgb_image = Image.open(rgb_path).convert('RGB')
        label_image = Image.open(label_path).convert('RGB')


        # augmentation
        if self.augment and random.random() < 0.5:
            rgb_image = TF.hflip(rgb_image)
            label_image = TF.hflip(label_image)

        label_logit = convert_rgb_to_logit(label_image)
        rgb_image = self.to_tensor(rgb_image)
        label_image = self.to_tensor(label_image)
        
        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, label_image, label_logit

color_map = {
    (255,255,255) : 0,      # undefined/background - white
    (178,176,153): 1,       # smooth trail - grey
    (128,255,0): 2,         # traversable grass - light green
    (156,76,30): 3,         # rough trail - brown
    (255,0,128): 4,         # puddle - pink
    (255,0,0): 5,           # obstacle - red
    (0,160,0): 6,           # non-traversable low vegetation - medium green
    (40,80,0): 7,           # high vegetation - dark green
    (1,88,255): 8,          # sky - blue
}

def convert_rgb_to_onehotencoding(label_img):
    label_np = np.array(label_img)  # (H, W, 3) - RGB
    h, w, _ = label_np.shape
    onehot = np.zeros((len(color_map), h, w), dtype=np.uint8) #uint8 as precised in the task

    for rgb, class_idx in color_map.items():
        mask = np.all(label_np == rgb, axis=-1)
        onehot[class_idx][mask] = 1

    return torch.from_numpy(onehot).float()

def convert_onehotencoding_to_rgb(onehot_tensor):
    assert onehot_tensor.shape[0] == len(color_map), "Expected shape (9, M, N)"

    class_map = onehot_tensor.argmax(dim=0).numpy()  # shape: (M, N) - get class index per pixel

    h, w = class_map.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8) #empty image size (M,N)=(h,w) in RGB (3 channels)

    inverse_color_map = {v: k for k, v in color_map.items()}
    for class_idx, color in inverse_color_map.items():
        rgb_image[class_map == class_idx] = color

    return Image.fromarray(rgb_image)

def convert_rgb_to_logit(label_img):
    label_np = np.array(label_img)  # (H, W, 3) - RGB
    h, w, _ = label_np.shape
    class_map = np.zeros((h, w), dtype=np.uint8) #uint8 as precised in the task

    for rgb, class_idx in color_map.items():
        mask = np.all(label_np == rgb, axis=-1)
        class_map[mask] = class_idx

    return torch.from_numpy(class_map)

def show_images_batch(images):
    grid = torchvision.utils.make_grid(images, nrow=2)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))  # convert (chanel, height, width) -> (height, width, channel)
    plt.axis('off')
    plt.show()
