import torch
<<<<<<< HEAD
from torch.utils.data import DataLoader, dataset
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from models.clip_model import CLIPModel
from tqdm import tqdm
import os
import requests
import zipfile
from PIL import Image
from torchvision import transforms


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized images"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def convert_to_rgb(img):
    """Standalone function to replace lambda for multiprocessing"""
    return img.convert('RGB') if img.mode != 'RGB' else img


def download_annotations():
    """Download and extract COCO annotations"""
    base_dir = os.path.expanduser("~/.cache/coco")
    annotations_dir = os.path.join(base_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = os.path.join(base_dir, "annotations.zip")

    if not os.path.exists(zip_path):
        print("Downloading annotations...")
        response = requests.get(annotations_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading annotations.zip",
                total=total_size,
                unit='iB',
                unit_scale=True,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)

def preprocess_dataset():
    """Preprocess COCO dataset with proper handling of variable-sized images"""
    try:
        # Setup dataset
        download_annotations()
        data_dir = os.path.expanduser("~/.cache/coco")
        ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
        img_dir = os.path.join(data_dir, "train2017")

        # Initialize CLIP model and preprocessing
        clip_model = CLIPModel()

        # Create transform without lambda functions
        transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),  # Use named function instead of lambda
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

        dataset = CocoDetection(
            root=img_dir,
            annFile=ann_file,
            transform=transform
        )

        # Use subset for demo
        dataset = torch.utils.data.Subset(dataset, range(10000))

        # Create dataloader with single worker to avoid pickling issues
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        # Process images
        all_features = []
        all_images = []

        print("Extracting image features...")
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Processing images"):
                # Stack images after ensuring they're the same size
                images = torch.stack(images).to(clip_model.device)
                features = clip_model.encode_image(images)
                all_features.append(features.cpu())
                all_images.append(images.cpu())

        # Save results
        os.makedirs("data", exist_ok=True)
        torch.save(torch.cat(all_features), "data/saved_features.pt")
        torch.save(torch.cat(all_images), "data/saved_images.pt")
        print("Preprocessing completed successfully!")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def verify_dataset(dataset):
    """Proper verification that works with Subsets"""
    print("\n=== DATASET VERIFICATION ===")

    # Access the original dataset if this is a Subset
    original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

    # Verify dataset type
    if isinstance(original_dataset, CocoDetection):
        print("✓ Dataset is COCO")
        print(f"Number of images: {len(dataset)}")

        # Access COCO API safely
        if hasattr(original_dataset, 'coco'):
            print("\nCOCO Dataset Info:")
            print(f"Categories: {len(original_dataset.coco.cats)}")
            print(f"Annotations: {len(original_dataset.coco.anns)}")
            print(f"Images: {len(original_dataset.coco.imgs)}")

            # Show sample image info
            img_id = original_dataset.ids[0]
            ann_ids = original_dataset.coco.getAnnIds(imgIds=[img_id])
            print(f"\nSample image ID: {img_id}")
            print(f"Associated annotations: {len(ann_ids)}")
        else:
            print("Warning: COCO API not initialized")
    else:
        print("✗ Dataset is NOT COCO")
        print(f"Actual type: {type(original_dataset).__name__}")


# Usage in preprocess_dataset():
verify_dataset(dataset)

if __name__ == "__main__":
    # Clear old features if they exist
    if os.path.exists("data/saved_features.pt"):
        os.remove("data/saved_features.pt")
    if os.path.exists("data/saved_images.pt"):
        os.remove("data/saved_images.pt")

=======
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
>>>>>>> 26dec6349fcce9d756c560c8358efbf46b65da81
    preprocess_dataset()