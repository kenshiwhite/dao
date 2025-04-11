import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from models.clip_model import CLIPModel
from tqdm import tqdm
import os


def preprocess_dataset():
    """Preprocess a dataset and save image features"""
    clip_model = CLIPModel()

    dataset = CIFAR100(
        root=os.path.expanduser("~/.cache"),
        download=True,
        transform=clip_model.preprocess
    )

    # Use subset for demo (remove for full dataset)
    dataset = torch.utils.data.Subset(dataset, range(1000))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract features
    all_features = []
    all_images = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Processing images"):
            images = images.to(clip_model.device)
            features = clip_model.encode_image(images)
            all_features.append(features.cpu())
            all_images.append(images.cpu())

    # Save results
    os.makedirs("data", exist_ok=True)
    torch.save(torch.cat(all_features), "data/saved_features.pt")
    torch.save(torch.cat(all_images), "data/saved_images.pt")


if __name__ == "__main__":
    preprocess_dataset()