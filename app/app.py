#app.py:
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.clip_model import CLIPModel
from utils.database import DatabaseManager

# Initialize components
clip_model = CLIPModel()
# At the top of app.py
from utils.database import DatabaseManager
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize database
#db = DatabaseManager()
#db.create_tables()  # Ensure tables exist


# Load precomputed features and images
def load_precomputed_data():
    features_path = "data/saved_features.pt"
    images_path = "data/saved_images.pt"

    if not (Path(features_path).exists() and Path(images_path).exists()):
        raise FileNotFoundError("Precomputed data not found. Run preprocessing first.")

    return torch.load(features_path), torch.load(images_path)


all_features, all_images = load_precomputed_data()


def create_classification_plot(probs, class_names):
    """Create a horizontal bar plot of classification probabilities"""
    plt.figure(figsize=(10, 5))
    y_pos = np.arange(len(probs))
    plt.barh(y_pos, probs, color='skyblue')
    plt.yticks(y_pos, class_names)
    plt.xlabel("Probability")
    plt.title("Zero-Shot Classification Results")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt

# Update the classify_image function to use COCO classes
def classify_image(image, class_names=None):
    """Classify an image using zero-shot classification"""
    if image is None:
        raise gr.Error("Please upload an image for classification.")

    # Use COCO classes if none provided
    if class_names is None:
        class_names = [
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

    probs = clip_model.zero_shot_classify(image, class_names)[0]
    top_indices = np.argsort(probs)[-5:][::-1]
    top_probs = probs[top_indices]
    top_classes = [class_names[i] for i in top_indices]

    plot = create_classification_plot(top_probs, top_classes)
    return plot

def search_images(query=None, query_image=None, top_k=4):
    """Search for similar images based on text or image query"""
    if query is None and query_image is None:
        raise gr.Error("Please provide either a text query or an image.")

    similarity = np.zeros(len(all_features))

    if query:
        text_features = clip_model.encode_text(query)
        similarity += clip_model.image_similarity(text_features, all_features)

    if query_image:
        image_features = clip_model.encode_image(query_image)
        similarity += clip_model.image_similarity(image_features, all_features)

    top_indices = np.argsort(similarity)[-top_k:][::-1]

    # Convert tensor images to PIL Images
    results = []
    for i in top_indices:
        img_tensor = all_images[i].clamp(0, 1)
        img_np = img_tensor.numpy().transpose(1, 2, 0)  # Convert CxHxW to HxWxC
        img_np = (img_np * 255).astype('uint8')
        results.append(Image.fromarray(img_np))

    # Save to database
#    for img in results:
#        db.save_query(query or "Image query", img)

    return results


def create_interface():
    with gr.Blocks(title="CLIP Image Search") as demo:
        gr.Markdown("# 🖼️ CLIP Image Search and Classification")

        with gr.Tab("Search"):
            with gr.Row():
                with gr.Column():
                    text_query = gr.Textbox(label="Text Query")
                    image_query = gr.Image(label="Image Query", type="pil")
                    search_btn = gr.Button("Search")
                with gr.Column():
                    results_gallery = gr.Gallery(label="Search Results")

            search_btn.click(
                search_images,
                inputs=[text_query, image_query],
                outputs=results_gallery
            )

        with gr.Tab("Classification"):
            with gr.Row():
                with gr.Column():
                    classify_input = gr.Image(label="Upload Image", type="pil")
                    classify_btn = gr.Button("Classify")
                with gr.Column():
                    classify_output = gr.Plot(label="Classification Results")

            classify_btn.click(
                fn=lambda img: classify_image(img),
                inputs=classify_input,
                outputs=classify_output
            )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()

