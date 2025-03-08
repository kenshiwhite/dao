import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
from models.clip_model import CLIPModel
from tqdm import tqdm
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_data():
    """Preprocess the dataset and save features and images."""
    clip_model = CLIPModel()

    # Load dataset (only first 7000 images)
    imagenet_v2_dataset = ImageNetV2Dataset(transform=clip_model.preprocess)
    imagenet_v2_dataset = torch.utils.data.Subset(imagenet_v2_dataset, range(7000))
    dataloader = DataLoader(imagenet_v2_dataset, batch_size=32, shuffle=False)

    # Store images and features
    all_images = []
    all_features = []
    print("Encoding all images...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, _ = batch
            images = images.to(clip_model.device).float()  # Ensure Float32 dtype
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_images.append(images.cpu())
            all_features.append(image_features.cpu())

    # Convert lists to tensors
    all_images = torch.cat(all_images)
    all_features = torch.cat(all_features)

    # Save features and images to disk
    os.makedirs("data", exist_ok=True)
    torch.save(all_features, "data/saved_features.pt")
    torch.save(all_images, "data/saved_images.pt")
    print("Features and images saved to disk for future use.")

if __name__ == "__main__":
    preprocess_data()