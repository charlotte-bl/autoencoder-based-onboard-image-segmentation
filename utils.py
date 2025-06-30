import numpy as np
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

class ImageSegmentationTrainingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # because of .DS_Store presence
        self.folders = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root_dir, self.folders[idx])

        rgb_path = os.path.join(folder_path, 'rgb.jpg')
        label_path = os.path.join(folder_path, 'labels.png')

        rgb_image = Image.open(rgb_path).convert('RGB')
        label_image = Image.open(label_path).convert('RGB')

        label_onehotencoding = convert_rgb_to_onehotencoding(label_image)

        if self.transform:
            rgb_image = self.transform(rgb_image)
            label_image = self.transform(label_image)

        return rgb_image, label_image, label_onehotencoding
    

# Labels format
color_map = {
    (1,88,255): 0,  # sky - blue
    (156,76,30): 1,      # rough trail - brown
    (178,176,153): 2,   # smooth trail - grey
    (128,255,0): 3,      # traversable grass - light green
    (40,80,0): 4,    # high vegetation - dark green
    (0,160,0): 5,      # non-traversable low vegetation - medium green
    (255,0,128): 6,     # puddle - pink
    (255,0,0): 7,  # obstacle - red
    (255,255,255) : 8, # undefined - white
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

def show_images_batch(images):
    grid = torchvision.utils.make_grid(images, nrow=2)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))  # convert (chanel, height, width) -> (height, width, channel)
    plt.axis('off')
    plt.show()