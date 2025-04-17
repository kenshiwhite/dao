import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from models.clip_model import CLIPModel
from tqdm import tqdm
import os
import requests
import zipfile
from PIL import Image
from torchvision import transforms

dataset = torch.utils.data.Subset(dataset, range(1000))
# In preprocess_data.py after dataset creation

print("\n=== DATASET VERIFICATION ===")
print(f"Dataset class: {type(dataset).__name__}")  # Should show CocoDetection
print(f"Number of images: {len(dataset)}")  # CIFAR=50k, COCO=118k train images
print(f"Sample annotation keys: {dataset.coco.dataset.keys()}")  # COCO-specific keys