import torch
import clip
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
import os
import datetime
import json
import logging

CONFIG = {
    "model_name": "ViT-B/32",
    "batch_size": 32,
    "epochs": 3,
    "lr": 5e-6,
    "image_size": 224,
    "coco_root": os.path.expanduser("~/.cache/coco"),
    "train_ann_file": "annotations/instances_train2017.json",
    "train_img_dir": "train2017",
    "log_dir": "logs",
    "checkpoint_dir": "checkpoints"
}


# Setup logging
def setup_logging():
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(CONFIG["log_dir"], f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class CocoCLIPDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        logger.info(f"Loaded COCO dataset with {len(self)} samples")

    def __getitem__(self, idx):
        try:
            img, targets = super().__getitem__(idx)
            img = img.convert("RGB")

            captions = [target['caption'] for target in targets if 'caption' in target]
            category_ids = [target['category_id'] for target in targets if 'category_id' in target]
            category_names = [self.categories[cat_id] for cat_id in category_ids]

            text = captions[0] if captions else f"a photo of {category_names[0]}" if category_names else ""
            return self.transform(img) if self.transform else img, text
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            return self.transform(Image.new('RGB', (224, 224))), ""


def get_transform(image_size):
    return Compose([
        Resize(image_size, interpolation=Image.BICUBIC),
        CenterCrop(image_size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711))
    ])


from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, epochs=3, lr=5e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    # Initialize SummaryWriter
    writer = SummaryWriter(CONFIG["log_dir"])
    logger.info(f"TensorBoard logging at {CONFIG['log_dir']}")

    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Training config: {json.dumps(CONFIG, indent=2)}")

    for epoch in range(epochs):
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (images, texts) in enumerate(progress):
            try:
                images = images.to(device)
                text_inputs = clip.tokenize(texts, truncate=True).to(device)

                logits_per_image, logits_per_text = model(images, text_inputs)
                labels = torch.arange(len(images)).to(device)

                loss = (criterion(logits_per_image, labels) + criterion(logits_per_text, labels)) / 2
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ✅ Log batch loss
                writer.add_scalar("Loss/Batch", loss.item(), epoch * len(dataloader) + batch_idx)

                progress.set_postfix({"loss": loss.item()})

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} completed. Avg loss: {avg_loss:.4f}")

        # ✅ Log epoch average loss
        writer.add_scalar("Loss/Epoch", avg_loss, epoch)

        checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    writer.close()  # ✅ Close the writer properly
    return model


if __name__ == "__main__":
    try:
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        os.makedirs(CONFIG["coco_root"], exist_ok=True)

        # Load model
        model, preprocess = clip.load(CONFIG["model_name"], device=device)
        model.train()
        logger.info(f"Loaded {CONFIG['model_name']} model")

        # Prepare dataset
        transform = get_transform(CONFIG["image_size"])
        train_dataset = CocoCLIPDataset(
            root=os.path.join(CONFIG["coco_root"], CONFIG["train_img_dir"]),
            annFile=os.path.join(CONFIG["coco_root"], CONFIG["train_ann_file"]),
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=0 if os.name == 'nt' else 4
        )
        logger.info(f"Created dataloader with {len(train_loader)} batches")

        # Train
        trained_model = train(model, train_loader, CONFIG["epochs"], CONFIG["lr"])

        # Save final model
        final_path = os.path.join(CONFIG["checkpoint_dir"], "final_model.pt")
        torch.save(trained_model.state_dict(), final_path)
        logger.info(f"Training complete. Model saved to {final_path}")

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)