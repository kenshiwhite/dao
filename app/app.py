import torch
import clip
import gradio as gr
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR100
import os
import matplotlib.pyplot as plt
from utils.helpers import is_english  # For English word validation
from utils.database import save_query_to_db, get_recent_queries, get_most_repeated_descriptions  # Database functions

# Set device and load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.eval()  # Keep model in evaluation mode

# Load CIFAR-100 dataset to get class labels
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True)
cifar100_classes = cifar100.classes  # Get the list of CIFAR-100 class names

# Load precomputed features and images
features_path = "data/saved_features.pt"
images_path = "data/saved_images.pt"

if os.path.exists(features_path) and os.path.exists(images_path):
    print("Loading precomputed features and images from disk...")
    all_features = torch.load(features_path)
    all_images = torch.load(images_path)
else:
    raise FileNotFoundError("Precomputed features and images not found. Run the preprocessing script first.")


def classify_image(image):
    """Classify an image using zero-shot classification."""
    try:
        # Validate image input
        if image is None:
            raise gr.Error("Please upload an image for classification.")

        image = preprocess(image).unsqueeze(0).to(device).float()

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Get text descriptions for CIFAR-100 classes
            text_descriptions = [f"This is a photo of a {label}" for label in cifar100_classes]
            text_tokens = clip.tokenize(text_descriptions).to(device)
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate probabilities
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

        # Prepare results for display
        top_probs = top_probs.squeeze().numpy()
        top_labels = top_labels.squeeze().numpy()

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        y = np.arange(len(top_probs))
        plt.barh(y, top_probs, color='skyblue')
        plt.yticks(y, [cifar100_classes[label] for label in top_labels])
        plt.xlabel("Probability")
        plt.title("Zero-Shot Classification Results")
        plt.gca().invert_yaxis()  # Invert y-axis to show the highest probability at the top
        plt.tight_layout()

        # Save the plot to a temporary file
        temp_file = "temp_plot.png"
        plt.savefig(temp_file, bbox_inches='tight')
        plt.close()

        # Return the plot as an image file path
        return temp_file

    except gr.Error as e:
        raise e  # Re-raise Gradio errors to show pop-up messages
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {str(e)}")


def search_images(query, uploaded_image=None):
    """Search for images based on text or image input and classify the top results."""
    try:
        # Validate query input
        if query and not is_english(query):
            raise gr.Error("Query must contain only English letters and spaces.")

        similarity = np.zeros(all_features.shape[0])  # Reset similarity on each call

        with torch.no_grad():
            if query:
                text_inputs = clip.tokenize([query]).to(device)
                text_features = model.encode_text(text_inputs).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity_text = (all_features @ text_features.T).squeeze().cpu().numpy()
                similarity += similarity_text  # Accumulate similarity scores

            if uploaded_image:
                try:
                    image = preprocess(uploaded_image).convert("RGB")
                    image = image.unsqueeze(0).to(device).float()
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity_image = (all_features @ image_features.T).squeeze().cpu().numpy()
                    similarity += similarity_image  # Add image similarity
                except Exception as e:
                    raise gr.Error("Invalid image. Please upload a valid image file.")

        sorted_indices = np.argsort(similarity)[::-1]
        top_images = [transforms.ToPILImage()(all_images[idx].clamp(0, 1)) for idx in sorted_indices[:4]]

        # Save the query to the database
        for image in top_images:
            image_path = f"data/query_results/{len(os.listdir('data/query_results')) + 1}.png"
            image.save(image_path)
            save_query_to_db(query or "Image-based query", image_path)

        # Perform zero-shot classification on the top images
        classification_results = []
        for image in top_images:
            classification_results.append(classify_image(image))

        return top_images, classification_results

    except gr.Error as e:
        raise e  # Re-raise Gradio errors to show pop-up messages
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {str(e)}")


def load_recent_queries():
    """Load and display recent queries from the database."""
    queries = get_recent_queries()
    images = [query[1] for query in queries]  # Extract image paths
    return images


def load_most_repeated_descriptions():
    """Load and display the most repeated descriptions and their photos."""
    queries = get_most_repeated_descriptions()
    images = [query[1] for query in queries]  # Extract image paths
    return images


# vremennyi gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ CLIP-based Image Search with Zero-Shot Classification")
    gr.Markdown(
        "Search for images using text or an example image, and see zero-shot classification results for the top matches!")

    with gr.Tab("Image Search"):
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="Enter Description", placeholder="e.g., 'a photo of a cat'")
                uploaded_image = gr.Image(type="pil", label="Upload Image (Optional)")
                search_button = gr.Button("Search")
            with gr.Column():
                top_results = gr.Gallery(label="Top Results")
                classification_results = gr.Gallery(label="Classification Results")

        search_button.click(
            fn=search_images,
            inputs=[query, uploaded_image],
            outputs=[top_results, classification_results]
        )

        most_repeated_queries = gr.Gallery(label="Most Repeated Queries")
        refresh_repeated_button = gr.Button("Refresh")

        refresh_repeated_button.click(
            fn=load_most_repeated_descriptions,
            outputs=most_repeated_queries
        )


    with gr.Tab("Zero-Shot Classification"):

        with gr.Row():
            with gr.Column():
                classify_image_input = gr.Image(type="pil", label="Upload Image for Classification")
                classify_button = gr.Button("Classify")
            with gr.Column():
                classification_output = gr.Image(type="filepath", label="Classification Results")

        classify_button.click(
            fn=classify_image,
            inputs=classify_image_input,
            outputs=classification_output
        )

    # with gr.Tab("Recent Queries"):
    #     recent_queries = gr.Gallery(label="Recent Queries")
    #     refresh_button = gr.Button("Refresh")
    #
    #     refresh_button.click(
    #         fn=load_recent_queries,
    #         outputs=recent_queries
    #     )

    gr.Markdown("### How It Works:")
    gr.Markdown("""
    1. **Image Search**: Enter a text description or upload an image to find similar images.
    2. **Zero-Shot Classification**: Upload an image to classify it using zero-shot classification.
    3. **Recent Queries**: View recently queried images and their descriptions.
    4. **Most Repeated Queries**: View the most frequently queried images and their descriptions.
    """)

# Launch the interface
demo.launch()