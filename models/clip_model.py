import torch
import clip
from torchvision import transforms
from PIL import Image

class CLIPModel:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        """Initialize the CLIP model."""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def preprocess_image(self, image):
        """Preprocess an image for CLIP."""
        return self.preprocess(image).unsqueeze(0).to(self.device).float()

    def encode_image(self, image):
        """Encode an image into feature vectors."""
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text):
        """Encode text into feature vectors."""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features