# app.py
import logging
import cv2
import numpy as np
import torch
import faiss
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from models.clip_model import CLIPModel
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from io import BytesIO
import base64
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)

class CLIPBackend:
    def __init__(self):
        self.clip_model = CLIPModel()
        self.all_features, self.all_images = self.load_precomputed_data()
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.index = faiss.IndexFlatL2(self.all_features.shape[1])
        self.index.add(self.all_features)

    def load_precomputed_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent
        features_path = base_dir / "scripts" / "data" / "saved_features.pt"
        images_path = base_dir / "scripts" / "data" / "saved_images.pt"

        if not features_path.exists() or not images_path.exists():
            raise FileNotFoundError("Precomputed data not found. Please check your paths.")

        features = torch.load(str(features_path))
        images = torch.load(str(images_path))
        return features, images

    def classify_image(self, image: Image.Image, class_names: Optional[List[str]] = None) -> Tuple[List[float], List[str]]:
        if image is None:
            raise ValueError("Please provide an image for classification.")

        class_names = class_names or self.coco_classes
        probs = self.clip_model.zero_shot_classify(image, class_names)[0]
        top_indices = np.argsort(probs)[-5:][::-1]
        top_probs = probs[top_indices]
        top_classes = [class_names[i] for i in top_indices]

        return top_probs.tolist(), top_classes

    async def search_images(self, query: Optional[str] = None, query_image: Optional[Image.Image] = None, top_k: int = 4) -> List[Tuple[torch.Tensor, float]]:
        if query is None and query_image is None:
            raise ValueError("Please provide either a text query or an image.")

        similarity = np.zeros(len(self.all_features))

        if query:
            text_features = self.clip_model.encode_text(query)
            similarity += self.clip_model.image_similarity(text_features, self.all_features)

        if query_image:
            image_features = self.clip_model.encode_image(query_image)
            similarity += self.clip_model.image_similarity(image_features, self.all_features)

        top_indices = np.argsort(similarity)[-top_k:][::-1]
        result_images = [self.all_images[i] for i in top_indices]
        result_scores = [similarity[i] for i in top_indices]

        return list(zip(result_images, result_scores))

    @staticmethod
    def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        img_tensor = img_tensor.clamp(0, 1)
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype('uint8')
        return Image.fromarray(img_np)

    def get_similarity_map(self, pil_img: Image.Image) -> np.ndarray:
        img_tensor = self.clip_model.preprocess(pil_img).unsqueeze(0).to(self.clip_model.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(img_tensor)
        similarity_map = features.squeeze().cpu().numpy()
        similarity_map = similarity_map.reshape(16, 32)  # Assuming 512-D features
        return similarity_map

    def generate_heatmap(self, image, similarity_map):
        """
        Generate a heatmap overlay on the input image using the similarity map.
        Args:
            image (PIL.Image): input image
            similarity_map (torch.Tensor or np.ndarray): similarity map (2D)
        Returns:
            PIL.Image: heatmap overlay
        """

        # Ensure similarity map is a numpy array
        if isinstance(similarity_map, torch.Tensor):
            similarity_map = similarity_map.detach().cpu().numpy()

        # Normalize similarity map to 0-1
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-6)

        # Resize similarity map to match image size
        similarity_map = Image.fromarray((similarity_map * 255).astype(np.uint8)).resize(image.size, resample=Image.BICUBIC)

        # Apply colormap
        cmap = plt.get_cmap('jet')
        colored_map = np.array(cmap(np.array(similarity_map)/255.0))[:, :, :3]  # Drop alpha channel

        # Convert back to PIL
        colored_map = (colored_map * 255).astype(np.uint8)
        heatmap = Image.fromarray(colored_map)

        # Blend the heatmap with the original image
        overlay = Image.blend(image.convert('RGB'), heatmap, alpha=0.5)

        return overlay

# Initialize FastAPI
app = FastAPI()
clip_backend = CLIPBackend()

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...), show_heatmap: bool = Form(False)):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        top_probs, top_classes = clip_backend.classify_image(image)

        response = {
            "top_probs": top_probs,
            "top_classes": top_classes,
        }

        if show_heatmap:
            heatmap = clip_backend.generate_heatmap(image)
            buffered = BytesIO()
            heatmap.save(buffered, format="JPEG")
            heatmap_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
            response["heatmap_base64"] = heatmap_base64

        return response

    except Exception as e:
        logging.error(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during classification.")

@app.post("/search_images")
async def search_images(query: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), show_heatmap: bool = Form(False)):
    try:
        if query:
            results = await clip_backend.search_images(query=query)
        elif file:
            image = Image.open(BytesIO(await file.read())).convert('RGB')
            results = await clip_backend.search_images(query_image=image)
        else:
            raise HTTPException(status_code=400, detail="Please provide either a text query or an image.")

        response = []

        for img_tensor, similarity_score in results:
            img_pil = clip_backend.tensor_to_pil(img_tensor)

            buffered = BytesIO()
            img_pil.save(buffered, format="JPEG")
            img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

            result = {
                "similarity": similarity_score,
                "image_base64": img_base64
            }

            if show_heatmap:
                heatmap = clip_backend.generate_heatmap(img_pil)
                heatmap_buffered = BytesIO()
                heatmap.save(heatmap_buffered, format="JPEG")
                heatmap_base64 = "data:image/jpeg;base64," + base64.b64encode(heatmap_buffered.getvalue()).decode('utf-8')
                result["heatmap_base64"] = heatmap_base64

            response.append(result)

        return {"results": response}

    except Exception as e:
        logging.error(f"Error searching images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during search.")
