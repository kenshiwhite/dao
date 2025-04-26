import logging
import numpy as np
import torch
import faiss
from PIL import Image
from pathlib import Path
from models.clip_model import CLIPModel
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import asyncio

# import redis
# import json

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

        # FAISS setup for fast retrieval
        self.index = faiss.IndexFlatL2(self.all_features.shape[1])
        self.index.add(self.all_features)

        # Redis caching setup (currently disabled)
        # self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

    def load_precomputed_data(self):
        """Load precomputed features and images"""
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent
        features_path = base_dir / "scripts" / "data" / "saved_features.pt"
        images_path = base_dir / "scripts" / "data" / "saved_images.pt"

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found at {features_path}")
        if not images_path.exists():
            raise FileNotFoundError(f"Images file not found at {images_path}")

        features = torch.load(str(features_path))
        images = torch.load(str(images_path))
        return features, images

    def classify_image(self, image: Image.Image, class_names: Optional[List[str]] = None) -> Tuple[
        List[float], List[str]]:
        """Classify an image using zero-shot classification"""
        if image is None:
            raise ValueError("Please provide an image for classification.")

        if class_names is None:
            class_names = self.coco_classes

        probs = self.clip_model.zero_shot_classify(image, class_names)[0]
        top_indices = np.argsort(probs)[-5:][::-1]
        top_probs = probs[top_indices]
        top_classes = [class_names[i] for i in top_indices]

        return top_probs.tolist(), top_classes

    async def search_images(self, query: str = None, query_image: Optional[Image.Image] = None, top_k: int = 4) -> List[
        Tuple[Image.Image, float]]:
        """Search for similar images based on text or image query, returning images and their similarity scores."""
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

        # Return list of (image tensor, similarity score) pairs
        return list(zip(result_images, result_scores))

    def tensor_to_pil(self, img_tensor):
        """Convert tensor image to PIL Image"""
        img_tensor = img_tensor.clamp(0, 1)
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype('uint8')
        return Image.fromarray(img_np)


app = FastAPI()

clip_backend = CLIPBackend()


@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    top_probs, top_classes = clip_backend.classify_image(image)
    return {"top_probs": top_probs, "top_classes": top_classes}


@app.post("/search_images")
async def search_images(query: Optional[str] = None, file: Optional[UploadFile] = File(None)):
    if query:
        results = await clip_backend.search_images(query=query)
    elif file:
        image = Image.open(BytesIO(await file.read()))
        results = await clip_backend.search_images(query_image=image)
    else:
        raise ValueError("Please provide either a query or an image.")

    # Prepare output: images are converted to base64 or skipped (depending on frontend plan)
    response = []
    for img_tensor, similarity_score in results:
        img_pil = clip_backend.tensor_to_pil(img_tensor)
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = "data:image/jpeg;base64," + img_bytes.hex()

        response.append({
            "similarity": similarity_score,
            "image_base64": img_base64
        })

    return {"results": response}
