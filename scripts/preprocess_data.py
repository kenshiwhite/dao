import os
import zipfile
import requests
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

from models.clip_model import CLIPModel


# ========== Utility Functions ==========

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized images"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def convert_to_rgb(img):
    """Convert non-RGB images to RGB"""
    return img.convert('RGB') if img.mode != 'RGB' else img


def download_file(url, dest_path, desc):
    """Generic file download with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))


def download_images():
    """Download and extract COCO images"""
    base_dir = os.path.expanduser("~/.cache/coco")
    img_zip_path = os.path.join(base_dir, "train2017.zip")
    img_dir = os.path.join(base_dir, "train2017")

    os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists(img_zip_path):
        print("Downloading COCO images...")
        download_file(
            url="http://images.cocodataset.org/zips/train2017.zip",
            dest_path=img_zip_path,
            desc="Downloading train2017.zip"
        )

    if not os.path.exists(os.path.join(img_dir, "000000000009.jpg")):
        print("Extracting COCO images...")
        with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)


def download_annotations():
    """Download and extract COCO annotations"""
    base_dir = os.path.expanduser("~/.cache/coco")
    ann_zip_path = os.path.join(base_dir, "annotations.zip")

    os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists(ann_zip_path):
        print("Downloading COCO annotations...")
        download_file(
            url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            dest_path=ann_zip_path,
            desc="Downloading annotations.zip"
        )

    print("Extracting COCO annotations...")
    with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)


def verify_dataset(dataset):
    """Verify dataset properties and COCO metadata"""
    print("\n=== DATASET VERIFICATION ===")

    original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

    if isinstance(original_dataset, CocoDetection):
        print("✓ Dataset is COCO")
        print(f"Number of samples: {len(dataset)}")

        if hasattr(original_dataset, 'coco'):
            print("\nCOCO Dataset Info:")
            print(f"Categories: {len(original_dataset.coco.cats)}")
            print(f"Annotations: {len(original_dataset.coco.anns)}")
            print(f"Images: {len(original_dataset.coco.imgs)}")

            img_id = original_dataset.ids[0]
            ann_ids = original_dataset.coco.getAnnIds(imgIds=[img_id])
            print(f"\nSample image ID: {img_id}")
            print(f"Associated annotations: {len(ann_ids)}")
        else:
            print("Warning: COCO API not initialized")
    else:
        print("✗ Dataset is NOT COCO")
        print(f"Actual type: {type(original_dataset).__name__}")


# ========== Main Preprocessing Pipeline ==========

def preprocess_dataset():
    """Preprocess COCO dataset and extract CLIP image features"""
    try:
        # Download dataset and annotations
        download_images()
        download_annotations()

        data_dir = os.path.expanduser("~/.cache/coco")
        ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
        img_dir = os.path.join(data_dir, "train2017")

        # Initialize CLIP model
        clip_model = CLIPModel()

        # Define transforms (no normalization yet for visualization purposes)
        transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create dataset and subset for demonstration
        dataset = CocoDetection(root=img_dir, annFile=ann_file, transform=transform)
        dataset = torch.utils.data.Subset(dataset, range(5000))

        # Verify dataset
        verify_dataset(dataset)

        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0
        )

        # Normalization (applied right before encoding)
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )

        all_features = []
        all_images = []

        print("\nExtracting image features...")
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Processing images"):
                # Stack and move to device
                image_batch = torch.stack(images).to(clip_model.device)

                # Normalize for CLIP encoding
                normalized_images = normalize(image_batch)

                # Encode features
                features = clip_model.encode_image(normalized_images)

                # Store features and original (unnormalized) images
                all_features.append(features.cpu())
                all_images.append(image_batch.cpu())

        # Save features and images
        os.makedirs("data", exist_ok=True)
        torch.save(torch.cat(all_features), "data/saved_features.pt")
        torch.save(torch.cat(all_images), "data/saved_images.pt")

        print("\nPreprocessing completed successfully!")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


# ========== Entry Point ==========

if __name__ == "__main__":
    # Remove previous saved files if they exist
    for file_path in ["data/saved_features.pt", "data/saved_images.pt"]:
        if os.path.exists(file_path):
            os.remove(file_path)

    preprocess_dataset()
