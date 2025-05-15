import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Paths
base_dir = Path(__file__).parent
images_path = base_dir / "scripts" / "data" / "saved_images.pt"

# Load
images = torch.load(images_path, weights_only=False)
print(f"Loaded {len(images)} images")

# Convert all to tensors (if any PILs exist)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

images_tensor = []

for img in images:
    if isinstance(img, Image.Image):
        img = preprocess(img)
    elif isinstance(img, torch.Tensor):
        pass  # already fine
    else:
        raise TypeError("Invalid image type:", type(img))
    images_tensor.append(img)

# Save cleaned
torch.save(images_tensor, images_path)
print("✅ Saved cleaned image list (as tensors only)")
