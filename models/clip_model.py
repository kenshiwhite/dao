import torch
import clip
import numpy as np
from torchvision import transforms
from PIL import Image


class CLIPModel:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_image(self, image):
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, text):
        with torch.no_grad():
            text_input = clip.tokenize([text]).to(self.device)
            features = self.model.encode_text(text_input)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def zero_shot_classify(self, image, class_names):

        """Perform zero-shot classification on an image with proper gradient handling"""
        with torch.no_grad():  # Ensure no gradients are tracked
            # Get image features
            image_features = self.encode_image(image)

            # Process text descriptions
            text_descriptions = [f"a photo of a {label}" for label in class_names]
            text_inputs = clip.tokenize(text_descriptions).to(self.device)
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Detach from computation graph before converting to numpy
            return logits.detach().cpu().numpy()

    def image_similarity(self, query_features, target_features):
        """Calculate similarity between query and target features"""
        with torch.no_grad():
            return (target_features @ query_features.T).squeeze().cpu().numpy()

        """Perform zero-shot classification on an image"""
        image_features = self.encode_image(image)

        text_descriptions = [f"a photo of a {label}" for label in class_names]
        text_inputs = clip.tokenize(text_descriptions).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return logits.cpu().numpy()

    def image_similarity(self, query_features, target_features):
        """Calculate similarity between query and target features"""
        return (target_features @ query_features.T).squeeze().cpu().numpy()

