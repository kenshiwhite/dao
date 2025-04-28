import torch
import clip
import numpy as np
from torchvision import transforms
from PIL import Image
from sentence_transformers import SentenceTransformer  # NEW
import cv2


class CLIPModel:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load vision CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Load multilingual text encoder
        self.text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

    def encode_image(self, image):
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, text):
        """Multilingual text encoding"""
        with torch.no_grad():
            text_features = self.text_model.encode([text], convert_to_tensor=True).to(self.device)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def zero_shot_classify(self, image, class_names):
        """Perform zero-shot classification on an image"""
        with torch.no_grad():
            image_features = self.encode_image(image)

            # Encode multilingual text descriptions
            text_features = self.text_model.encode(
                [f"a photo of a {label}" for label in class_names],
                convert_to_tensor=True
            ).to(self.device)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return logits.cpu().numpy()

    def image_similarity(self, query_features, target_features):
        """Calculate similarity between query and target features"""
        with torch.no_grad():
            return (target_features @ query_features.T).squeeze().cpu().numpy()

    def get_last_selfattention(self, image: Image.Image):
        """Extract the last self-attention map from CLIP ViT"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # forward manually through visual encoder
        x = self.model.visual.conv1(image_input)  # shape = [batch_size, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # flatten grid
        x = x.permute(0, 2, 1)  # shape = [batch_size, num_patches, width]
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype).expand(x.shape[0], -1, -1), x],
            dim=1,
        )  # prepend class token
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        # record attentions
        attentions = []
        for blk in self.model.visual.transformer.resblocks:
            x, attn = blk(x, return_attention=True)
            attentions.append(attn)

        return attentions[-1]  # last block's attention
