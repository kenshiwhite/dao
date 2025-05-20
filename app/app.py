from fastapi import status, FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple
import logging
import torch
import numpy as np
import faiss
import time
from io import BytesIO
from pathlib import Path
from PIL import Image
import base64
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from models.clip_model import CLIPModel
from utils.database import Database

# Set up logging
logging.basicConfig(level=logging.INFO)

class CLIPBackend:
    def __init__(self):
        self.clip_model = CLIPModel()
        self.all_features, self.all_images = self.load_precomputed_data()
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        self.index = faiss.IndexFlatL2(self.all_features.shape[1])
        self.index.add(self.all_features)

    def add_image_to_index(self, image: Image.Image):
        with torch.no_grad():
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image_tensor = transform(image)
            normalize = transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
            normalized_tensor = normalize(image_tensor).unsqueeze(0).to(self.clip_model.device)
            image_feature = self.clip_model.encode_image(normalized_tensor).cpu()
            image_tensor_for_storage = image_tensor

        self.all_features = torch.cat([self.all_features, image_feature], dim=0)
        self.all_images.append(image_tensor_for_storage)
        self.index.add(image_feature.numpy())

        script_dir = Path(__file__).parent
        base_dir = script_dir.parent
        features_path = base_dir / "data" / "saved_features.pt"
        images_path = base_dir / "data" / "saved_images.pt"

        torch.save(self.all_features, features_path)
        torch.save(self.all_images, images_path)

        logging.info("New image added to index and saved.")

    def load_precomputed_data(self) -> Tuple[torch.Tensor, List[Image.Image]]:
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent
        features_path = base_dir / "data" / "saved_features.pt"
        images_path = base_dir / "data" / "saved_images.pt"

        if not features_path.exists() or not images_path.exists():
            raise FileNotFoundError("Precomputed data not found. Please check your paths.")

        features = torch.load(str(features_path), weights_only=False)
        images = torch.load(str(images_path), weights_only=False)

        if isinstance(images, torch.Tensor):
            images = list(images)
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


# Initialize app and dependencies
app = FastAPI()
clip_backend = CLIPBackend()
security = HTTPBasic()
db = Database()

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> int:
    user_id, role = db.authenticate_user(credentials.username, credentials.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    try:
        return int(user_id)
    except ValueError:
        logging.error(f"Invalid user_id: expected int, got '{user_id}'")
        raise HTTPException(status_code=500, detail="Internal server error: invalid user ID")

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> int:
    user_id, role = db.authenticate_user(credentials.username, credentials.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return int(user_id)

@app.post("/admin/upload_image", status_code=status.HTTP_201_CREATED)
async def admin_upload_image(file: UploadFile = File(...), user_id: int = Depends(verify_admin)):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        clip_backend.add_image_to_index(image)
        return {"detail": "Image uploaded and added to index."}
    except Exception as e:
        logging.error(f"Admin upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload and index image.")

@app.get("/recent_queries")
async def get_recent_queries(user_id: int = Depends(get_current_user)):
    try:
        recent_queries = db.get_recent_queries(user_id)
        return {"recent_queries": recent_queries}
    except Exception as e:
        logging.error(f"Error fetching recent queries: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching recent queries.")

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...), user_id: int = Depends(get_current_user)):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        top_probs, top_classes = clip_backend.classify_image(image)
        image_path = f"classification_user{user_id}_{int(time.time())}.jpg"
        image.save(image_path)
        db.save_classification(user_id, image_path, top_classes, top_probs, int(time.time()))
        return {
            "top_probs": top_probs,
            "top_classes": top_classes
        }
    except Exception as e:
        logging.error(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during classification.")

class FeedbackRequest(BaseModel):
    feedback_text: str

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, user_id: int = Depends(get_current_user)):
    try:
        db.save_feedback(user_id, feedback.feedback_text)
        return {"message": "Thank you for your feedback!"}
    except Exception as e:
        logging.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving feedback.")

@app.get("/feedback")
async def get_feedback(user_id: int = Depends(get_current_user)):
    try:
        feedbacks = db.get_feedbacks(user_id)
        return {"feedbacks": feedbacks}
    except Exception as e:
        logging.error(f"Error retrieving feedbacks: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching feedbacks.")

@app.get("/recent_classifications")
async def get_recent_classifications(user_id: int = Depends(get_current_user)):
    try:
        classifications = db.get_recent_classifications(user_id)
        return {"recent_classifications": classifications}
    except Exception as e:
        logging.error(f"Error fetching recent classifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching classifications.")

@app.get("/top-queries")
def get_most_searched_queries(limit: int = 3):
    try:
        top_queries: List[Tuple[str, int]] = db.get_top_queries(limit=limit)
        return JSONResponse(content={
            "top_queries": [{"query": q, "count": c} for q, c in top_queries]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/search_images")
async def search_images(
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
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
            db.save_query(query_text=query_text, image_path=image_path, user_id=user_id)

            result = {
                "similarity": similarity_score,
                "image_base64": img_base64
            }
            response.append(result)

        return {"results": response}
    except Exception as e:
        logging.error(f"Error searching images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during search.")
