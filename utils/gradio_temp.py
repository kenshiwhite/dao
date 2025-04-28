import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from app.app import CLIPBackend
from PIL import Image
import asyncio

backend = CLIPBackend()

def create_classification_plot(probs, class_names):
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(probs))
    ax.barh(y_pos, probs, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Probability")
    ax.set_title("Zero-Shot Classification Results")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def classify_image(image, class_names=None):
    if image is None:
        raise gr.Error("Please upload an image for classification.")

    probs, classes = backend.classify_image(image, class_names)
    plot = create_classification_plot(probs, classes)

    similarity_map = backend.get_similarity_map(image)
    heatmap = backend.generate_heatmap(image, similarity_map)

    return plot, heatmap

async def search_images(query=None, query_image=None, top_k=4):
    """Wrapper for image search with Gradio."""
    if not query and query_image is None:
        raise gr.Error("Please provide either a text query or an image.")

    results = await backend.search_images(query=query, query_image=query_image, top_k=top_k)

    images_with_labels = []
    heatmaps = []

    for img_tensor, score in results:
        pil_img = backend.tensor_to_pil(img_tensor)
        label = f"Similarity: {score:.2f}"

        similarity_map = backend.get_similarity_map(pil_img)
        heatmap = backend.generate_heatmap(pil_img, similarity_map)

        images_with_labels.append((pil_img, label))
        heatmaps.append(heatmap)

    return images_with_labels, heatmaps

def create_interface():
    with gr.Blocks(title="CLIP Image Search and Classification with Heatmap") as demo:
        gr.Markdown("# 🖼️ CLIP Image Search and Classification with Heatmap")

        with gr.Tab("Search"):
            with gr.Row():
                with gr.Column():
                    text_query = gr.Textbox(label="Text Query")
                    image_query = gr.Image(label="Image Query", type="pil")
                    search_btn = gr.Button("Search")
                with gr.Column():
                    results_gallery = gr.Gallery(label="Search Results", columns=[2], object_fit="contain", allow_preview=True)
                    heatmap_gallery = gr.Gallery(label="Heatmaps", columns=[2], object_fit="contain", allow_preview=True)

            search_btn.click(
                fn=search_images,
                inputs=[text_query, image_query],
                outputs=[results_gallery, heatmap_gallery]
            )

        with gr.Tab("Classification"):
            with gr.Row():
                with gr.Column():
                    classify_input = gr.Image(label="Upload Image", type="pil")
                    classify_btn = gr.Button("Classify")
                with gr.Column():
                    classify_output_plot = gr.Plot(label="Classification Results")
                    classify_output_heatmap = gr.Image(label="Generated Heatmap", type="pil")

            classify_btn.click(
                fn=classify_image,
                inputs=[classify_input],
                outputs=[classify_output_plot, classify_output_heatmap]
            )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(show_api=False)
