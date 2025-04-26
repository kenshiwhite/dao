import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from app.app import CLIPBackend

backend = CLIPBackend()


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


def classify_image(image, class_names=None):
    """Wrapper for classification with Gradio"""
    if image is None:
        raise gr.Error("Please upload an image for classification.")

    probs, classes = backend.classify_image(image, class_names)
    return create_classification_plot(probs, classes)


async def search_images(query=None, query_image=None, top_k=4):
    """Wrapper for image search with Gradio"""
    if query is None and query_image is None:
        raise gr.Error("Please provide either a text query or an image.")

    # Now backend.search_images returns (images, scores)
    results = await backend.search_images(query=query, query_image=query_image, top_k=top_k)

    images_with_scores = []
    for img, score in results:
        pil_img = backend.tensor_to_pil(img)
        label = f"Similarity: {score:.2f}"
        images_with_scores.append((pil_img, label))

    return images_with_scores


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
                    results_gallery = gr.Gallery(label="Search Results", columns=2, height="auto")


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
                classify_image,
                inputs=classify_input,
                outputs=classify_output
            )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
