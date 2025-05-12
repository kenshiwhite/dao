import logging
import time
import torch
import numpy as np
import faiss
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from models.clip_model import CLIPModel
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from io import BytesIO
import base64
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from utils.database import Database

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

    async def search_images(self, query: Optional[str] = None, query_image: Optional[Image.Image] = None, top_k: int = 10) -> List[Tuple[torch.Tensor, float]]:
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
        if isinstance(similarity_map, torch.Tensor):
            similarity_map = similarity_map.detach().cpu().numpy()

        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-6)
        similarity_map = Image.fromarray((similarity_map * 255).astype(np.uint8)).resize(image.size, resample=Image.BICUBIC)

        cmap = plt.get_cmap('jet')
        colored_map = np.array(cmap(np.array(similarity_map) / 255.0))[:, :, :3]
        colored_map = (colored_map * 255).astype(np.uint8)
        heatmap = Image.fromarray(colored_map)

        overlay = Image.blend(image.convert('RGB'), heatmap, alpha=0.5)
        return overlay

# Initialize FastAPI
app = FastAPI()
clip_backend = CLIPBackend()
security = HTTPBasic()
db = Database()

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> int:
    user_id, role = db.authenticate_user(credentials.username, credentials.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user_id

@app.get("/recent_queries")
async def get_recent_queries(user_id: int = Depends(get_current_user)):
    try:
        recent_queries = db.get_recent_queries(user_id)
        return {"recent_queries": recent_queries}
    except Exception as e:
        logging.error(f"Error fetching recent queries: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching recent queries.")

@app.post("/classify_image")
async def classify_image(
    file: UploadFile = File(...),
    show_heatmap: bool = Form(False),
    user_id: int = Depends(get_current_user)
):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        top_probs, top_classes = clip_backend.classify_image(image)

        response = {
            "top_probs": top_probs,
            "top_classes": top_classes,
        }

        if show_heatmap:
            similarity_map = clip_backend.get_similarity_map(image)
            heatmap = clip_backend.generate_heatmap(image, similarity_map)
            buffered = BytesIO()
            heatmap.save(buffered, format="JPEG")
            heatmap_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
            response["heatmap_base64"] = heatmap_base64

        db.save_classification(user_id, top_probs, top_classes)

        return response

    except Exception as e:
        logging.error(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during classification.")

@app.get("/recent_classifications")
async def get_recent_classifications(user_id: int = Depends(get_current_user)):
    try:
        classifications = db.get_recent_classifications(user_id)
        return {"recent_classifications": classifications}
    except Exception as e:
        logging.error(f"Error fetching recent classifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching classifications.")

@app.post("/search_images")
async def search_images(
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
    show_heatmap: bool = Form(False),
    user_id: int = Depends(get_current_user)
):
    try:
        if query:
            results = await clip_backend.search_images(query=query, top_k=top_k)
            query_text = query
        elif file:
            image = Image.open(BytesIO(await file.read())).convert('RGB')
            results = await clip_backend.search_images(query_image=image, top_k=top_k)
            query_text = "[IMAGE]"
        else:
            raise HTTPException(status_code=400, detail="Please provide either a text query or an image.")

        response = []

        for idx, (img_tensor, similarity_score) in enumerate(results):
            img_pil = clip_backend.tensor_to_pil(img_tensor)
            buffered = BytesIO()
            img_pil.save(buffered, format="JPEG")
            img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

            image_path = f"search_result_user{user_id}_{int(time.time())}_{idx}.jpg"
            img_pil.save(image_path)

            # Сохраняем запрос в таблицу search_logs
            db.save_query(query_text=query_text, image_path=image_path, user_id=user_id)

            result = {
                "similarity": similarity_score,
                "image_base64": img_base64
            }

            if show_heatmap:
                similarity_map = clip_backend.get_similarity_map(img_pil)
                heatmap = clip_backend.generate_heatmap(img_pil, similarity_map)
                heatmap_buffered = BytesIO()
                heatmap.save(heatmap_buffered, format="JPEG")
                heatmap_base64 = "data:image/jpeg;base64," + base64.b64encode(heatmap_buffered.getvalue()).decode('utf-8')
                result["heatmap_base64"] = heatmap_base64

            response.append(result)

        return {"results": response}

    except Exception as e:
        logging.error(f"Error searching images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during search.")
