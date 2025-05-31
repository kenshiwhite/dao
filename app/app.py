from fastapi import status, FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPBasicCredentials, HTTPBasic
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
from pydantic import BaseModel
from models.clip_model import CLIPModel
from utils.database import Database
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from datetime import date
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours


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

    def classify_image(self, image: Image.Image, class_names: Optional[List[str]] = None) -> Tuple[
        List[float], List[str]]:
        if image is None:
            raise ValueError("Please provide an image for classification.")

        class_names = class_names or self.coco_classes
        probs = self.clip_model.zero_shot_classify(image, class_names)[0]
        top_indices = np.argsort(probs)[-5:][::-1]
        top_probs = probs[top_indices]
        top_classes = [class_names[i] for i in top_indices]

        return top_probs.tolist(), top_classes

    async def search_images(self, query: Optional[str] = None, query_image: Optional[Image.Image] = None,
                            top_k: int = 10) -> List[Tuple[torch.Tensor, float]]:
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

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Authorization", "Content-Type", "Accept"],
    expose_headers=["*"]
)

clip_backend = CLIPBackend()
security = HTTPBearer()
basic_security = HTTPBasic()  # Keep for backward compatibility if needed
db = Database()


# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        username: str = payload.get("username")
        role: str = payload.get("role")
        if user_id is None:
            return None, None, None
        return user_id, username, role
    except JWTError as e:
        logging.error(f"JWT Error: {str(e)}")
        return None, None, None


# Pydantic models
class UserRegistration(BaseModel):
    username: str
    password: str
    role: str = "user"


class UserLogin(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    message: str
    user_id: int
    username: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_id: int
    username: str
    role: str


class DeleteFeedbackRequest(BaseModel):
    feedback_id: int


class FeedbackRequest(BaseModel):
    feedback_text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

# Authentication dependencies
def get_current_user(token: str = Depends(security)) -> dict:
    """Get current user from JWT token"""
    try:
        user_id, username, role = verify_token(token.credentials)
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {
            "user_id": user_id,
            "username": username,
            "role": role
        }
    except Exception as e:
        logging.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Verify user has admin privileges"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    return current_user


# Authentication endpoints
@app.post("/api/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegistration):
    """Register a new user"""
    try:
        # Check if username already exists
        existing_user, _ = db.authenticate_user(user_data.username, "dummy_password")
        if existing_user is not None:
            raise HTTPException(
                status_code=400,
                detail="Username already exists"
            )

        # Register the new user
        db.register_user(user_data.username, user_data.password, user_data.role)

        # Get the newly created user's ID
        user_id, role = db.authenticate_user(user_data.username, user_data.password)

        if user_id is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to create user"
            )

        return AuthResponse(
            message="User registered successfully",
            user_id=user_id,
            username=user_data.username,
            role=role
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during registration"
        )


@app.post("/api/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin):
    """Authenticate a user and return JWT token"""
    try:
        user_id, role = db.authenticate_user(login_data.username, login_data.password)

        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )

        # Create JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "user_id": user_id,
                "username": login_data.username,
                "role": role
            },
            expires_delta=access_token_expires
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user_id,
            username=login_data.username,
            role=role
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during login"
        )


@app.get("/api/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user's profile information"""
    try:
        user_id = current_user["user_id"]

        # Get additional user data from database
        user_data = db.execute_query(
            "SELECT username, created_at, last_login FROM users WHERE id = %s",
            (user_id,),
            fetch=True
        )

        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        username, created_at, last_login = user_data[0]

        return {
            "user_id": user_id,
            "username": username,
            "role": current_user["role"],
            "created_at": created_at,
            "last_login": last_login
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Profile error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching user profile"
        )


# Admin endpoints
@app.post("/api/admin/upload_image", status_code=status.HTTP_201_CREATED)
async def admin_upload_image(
        file: UploadFile = File(...),
        current_user: dict = Depends(verify_admin)
):
    """Admin endpoint to upload and index images"""
    try:
        # Validate file
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )

        # Read and validate file content
        file_contents = await file.read()
        if len(file_contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(file_contents) > 20 * 1024 * 1024:  # 20MB limit for admin
            raise HTTPException(status_code=400, detail="File too large (max 20MB)")

        # Process image
        try:
            image = Image.open(BytesIO(file_contents)).convert('RGB')
        except Exception as e:
            logging.error(f"Error processing uploaded image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image")

        # Add to index
        try:
            clip_backend.add_image_to_index(image)
        except Exception as e:
            logging.error(f"Error adding image to index: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to add image to search index")

        return {
            "status": "success",
            "message": "Image uploaded and added to index successfully",
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Admin upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")

# User query and classification endpoints
@app.get("/api/recent_queries")
async def get_recent_queries(current_user: dict = Depends(get_current_user)):
    """Get recent queries for the current user as an array of strings"""
    try:
        user_id = current_user["user_id"]
        logging.info(f"Fetching recent queries for user_id: {user_id}")

        # Use the database method directly - it's already correct
        recent_queries = db.get_recent_queries(user_id, limit=10)

        logging.info(f"Retrieved queries: {recent_queries}")

        return {
            "recent_queries": recent_queries,
            "status": "success"
        }

    except Exception as e:
        import traceback
        logging.error(f"Error fetching recent queries: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")

        # Return empty array instead of raising exception
        return {
            "recent_queries": [],
            "status": "error",
            "message": str(e)
        }


@app.post("/api/classify_image")
async def classify_image(
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user)
):
    """Classify an uploaded image"""
    try:
        # Enhanced validation
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check file size (limit to 10MB)
        file_contents = await file.read()
        if len(file_contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        if len(file_contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Validate file type more thoroughly
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            )

        user_id = current_user["user_id"]
        logging.info(f"Processing image classification for user_id: {user_id}")

        # Process the image with better error handling
        try:
            image = Image.open(BytesIO(file_contents)).convert('RGB')
            logging.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            logging.error(f"Error opening image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        # Classify the image
        try:
            top_probs, top_classes = clip_backend.classify_image(image)
            logging.info(f"Classification successful: {top_classes[:3]}")  # Log top 3 classes
        except Exception as e:
            logging.error(f"Error during classification: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image classification")

        # Save the classification results (optional - don't fail if this fails)
        try:
            image_path = f"classification_user{user_id}_{int(time.time())}.jpg"
            # Create directory if it doesn't exist
            import os
            os.makedirs("uploads", exist_ok=True)
            full_path = os.path.join("uploads", image_path)
            image.save(full_path)

            db.save_classification(user_id, image_path, top_classes, top_probs, int(time.time()))
            logging.info(f"Classification saved to database: {image_path}")
        except Exception as e:
            logging.warning(f"Could not save classification to database: {str(e)}")
            # Continue without failing - the classification still worked

        return {
            "status": "success",
            "top_probs": top_probs,
            "top_classes": top_classes,
            "message": "Image classified successfully",
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in classify_image: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image classification"
        )


# Feedback endpoints
@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        db.save_feedback(user_id=user_id, feedback_text=feedback.feedback_text)
        return {"message": "Thank you for your feedback!"}
    except Exception as e:
        logging.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving feedback.")


@app.delete("/api/feedback/{feedback_id}")
async def delete_feedback(
        feedback_id: int,
        current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["user_id"]
        is_admin = current_user["role"] == "admin"

        # Try to delete the feedback
        success = db.delete_feedback(feedback_id, user_id, is_admin)

        if success:
            return {"message": "Feedback deleted successfully"}
        else:
            if is_admin:
                raise HTTPException(status_code=404, detail="Feedback not found")
            else:
                raise HTTPException(status_code=403, detail="You can only delete your own feedback")

    except Exception as e:
        logging.error(f"Error deleting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting feedback")

@app.get("/api/feedback")
async def get_feedback(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        feedbacks = db.get_feedbacks(user_id)
        return feedbacks
    except Exception as e:
        logging.error(f"Error retrieving feedbacks: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching feedbacks.")


@app.get("/api/recent_classifications")
async def get_recent_classifications(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        classifications = db.get_recent_classifications(user_id)
        return {"recent_classifications": classifications}
    except Exception as e:
        logging.error(f"Error fetching recent classifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching classifications.")


@app.get("/api/top-queries")
def get_most_searched_queries(limit: int = 3):
    try:
        top_queries: List[Tuple[str, int]] = db.get_top_queries(limit=limit)
        return JSONResponse(content={
            "top_queries": [{"query": q, "count": c} for q, c in top_queries]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/search_images")
async def search_images(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["user_id"]

        # Validate that query is not empty
        if not request.query or request.query.strip() == "":
            raise HTTPException(status_code=400, detail="Please provide a text query.")

        # Search using text query only
        results = await clip_backend.search_images(query=request.query.strip(), top_k=request.top_k)
        query_text = request.query.strip()

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
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error searching images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during search.")


@app.get("/api/analytics/today")
async def get_today_analytics(current_user: dict = Depends(get_current_user)):
    """Get analytics for today's search queries (no personal data exposed)"""
    try:
        # Analytics available to all authenticated users
        today = date.today()

        # Get today's queries from database - Fixed the column name issue
        today_queries = db.execute_query(
            """
            SELECT query_text, timestamp 
            FROM queries 
            WHERE DATE(timestamp) = %s 
            AND query_text IS NOT NULL 
            AND query_text != ''
            """,
            (today,),
            fetch=True
        )

        if not today_queries:
            return {
                "date": today.isoformat(),
                "total_searches": 0,
                "unique_queries": 0,
                "top_queries": [],
                "hourly_distribution": [],
                "query_length_stats": {
                    "avg_length": 0,
                    "min_length": 0,
                    "max_length": 0
                }
            }

        # Process the queries - Added null checks
        queries_text = []
        queries_time = []

        for query in today_queries:
            if query[0] and query[0].strip():  # Check if query_text is not None and not empty
                queries_text.append(query[0].strip().lower())
                queries_time.append(query[1])

        # If no valid queries after filtering
        if not queries_text:
            return {
                "date": today.isoformat(),
                "total_searches": 0,
                "unique_queries": 0,
                "top_queries": [],
                "hourly_distribution": [],
                "query_length_stats": {
                    "avg_length": 0,
                    "min_length": 0,
                    "max_length": 0
                }
            }

        # Calculate statistics
        total_searches = len(queries_text)
        unique_queries = len(set(queries_text))

        # Top queries (most frequent) - Only include queries with count > 0 (which they all will be)
        query_counter = Counter(queries_text)
        top_queries = [
            {"query": query, "count": count}
            for query, count in query_counter.most_common(10)
            if count > 0  # This ensures only non-zero counts (though all will be > 0 anyway)
        ]

        # Hourly distribution - Only include hours with searches
        hourly_counts = {}
        for query_time in queries_time:
            try:
                # Handle both datetime and timestamp formats
                if hasattr(query_time, 'hour'):
                    hour = query_time.hour
                else:
                    # If it's a timestamp, convert it
                    from datetime import datetime
                    if isinstance(query_time, (int, float)):
                        query_time = datetime.fromtimestamp(query_time)
                    hour = query_time.hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            except Exception as e:
                logging.warning(f"Error processing query time {query_time}: {str(e)}")
                continue

        # Only include hours that have searches (count > 0)
        hourly_distribution = [
            {"hour": hour, "count": count}
            for hour, count in hourly_counts.items()
            if count > 0
        ]

        # Sort by hour for better readability
        hourly_distribution.sort(key=lambda x: x["hour"])

        # Query length statistics - Added safety checks
        query_lengths = []
        for query in queries_text:
            try:
                length = len(query.split()) if query else 0
                query_lengths.append(length)
            except Exception:
                query_lengths.append(0)

        if query_lengths:
            avg_length = sum(query_lengths) / len(query_lengths)
            min_length = min(query_lengths)
            max_length = max(query_lengths)
        else:
            avg_length = min_length = max_length = 0

        return {
            "date": today.isoformat(),
            "total_searches": total_searches,
            "unique_queries": unique_queries,
            "top_queries": top_queries,
            "hourly_distribution": hourly_distribution,
            "query_length_stats": {
                "avg_length": round(avg_length, 2),
                "min_length": min_length,
                "max_length": max_length
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching analytics: {str(e)}")
        # Log more detailed error information for debugging
        import traceback
        logging.error(f"Analytics error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error fetching analytics data")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# Root endpoint
@app.get("/")
async def root():
    return {"message": "CLIP Image Search API", "version": "1.0.0"}