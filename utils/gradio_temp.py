import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from app.app import CLIPBackend
from PIL import Image
import asyncio
from utils.database import Database

# Initialize backend and DB
backend = CLIPBackend()
db = Database()
db.create_tables()


# Helper to plot classification results
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


# Register new user
def register_user(username, password):
    if not username or not password:
        return "❌ Username and password cannot be empty."
    try:
        db.register_user(username, password)
        return f"✅ User '{username}' registered successfully!"
    except Exception as e:
        if 'unique constraint' in str(e).lower():
            return f"⚠️ Username '{username}' is already taken."
        return f"❌ Registration failed: {str(e)}"


# Login user
def login_user(username, password):
    user_id, role = db.authenticate_user(username, password)
    if user_id is None:
        return gr.update(visible=False), gr.update(visible=False), "❌ Invalid credentials.", None, None
    return gr.update(visible=True), gr.update(visible=True), f"✅ Welcome, {username}!", user_id, username


# Classification function
def classify_image(image, user_id, username, class_names=None):
    if user_id is None:
        raise gr.Error("🔒 Please log in to classify images.")
    if image is None:
        raise gr.Error("🖼️ Please upload an image.")

    probs, classes = backend.classify_image(image, class_names)
    plot = create_classification_plot(probs, classes)
    similarity_map = backend.get_similarity_map(image)
    heatmap = backend.generate_heatmap(image, similarity_map)

    db.save_query("Image Classification", "image_classification.png", user_id)
    return plot, heatmap


# Search function
def search_images_wrapper(query, query_image, top_k, user_id, username):
    return asyncio.run(search_images(query, query_image, top_k, user_id, username))


async def search_images(query, query_image, top_k, user_id, username):
    if user_id is None:
        raise gr.Error("🔒 Please log in to perform search.")
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

    db.save_query(query if query else "Image Search", "search_image.png", user_id)
    return images_with_labels, heatmaps


# Suggest dropdown
def suggest_queries():
    recent = db.get_recent_queries(limit=5)
    return gr.Dropdown.update(choices=[q[0] for q in recent], visible=True) if recent else gr.Dropdown.update(
        visible=False)


def fill_textbox(choice):
    return gr.Textbox.update(value=choice)


# Interface definition
def create_interface():
    with gr.Blocks(title="CLIP Auth Image Search") as demo:
        gr.Markdown("# 🖼️ CLIP Image Search & Classification with 🔐 Authentication")

        # --- Login/Register ---
        with gr.Row():
            login_username = gr.Textbox(label="Username")
            login_password = gr.Textbox(label="Password", type="password")
        with gr.Row():
            login_btn = gr.Button("Login")
            register_btn = gr.Button("Register")
        login_msg = gr.Textbox(label="Status", interactive=False)

        # State
        user_id_state = gr.State()
        username_state = gr.State()

        # --- Main Functional UI (Initially Hidden) ---
        with gr.Row(visible=False) as search_section:
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(label="Text Query")
                        dropdown = gr.Dropdown(choices=[], label="Suggestions", interactive=True, visible=False)
                        image_query = gr.Image(label="Image Query", type="pil")
                        top_k_input = gr.Number(value=4, label="Top K")
                        search_btn = gr.Button("Search")
                    with gr.Column():
                        results_gallery = gr.Gallery(label="Search Results", columns=[2], object_fit="contain",
                                                     allow_preview=True)
                        heatmap_gallery = gr.Gallery(label="Heatmaps", columns=[2], object_fit="contain",
                                                     allow_preview=True)

                text_input.focus(fn=suggest_queries, inputs=[], outputs=[dropdown])
                dropdown.change(fn=fill_textbox, inputs=[dropdown], outputs=[text_input])
                dropdown.change(fn=lambda: gr.Dropdown.update(visible=False), inputs=[], outputs=[dropdown])

                search_btn.click(
                    fn=search_images_wrapper,
                    inputs=[text_input, image_query, top_k_input, user_id_state, username_state],
                    outputs=[results_gallery, heatmap_gallery]
                )

        with gr.Row(visible=False) as classify_section:
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
                    inputs=[classify_input, user_id_state, username_state],
                    outputs=[classify_output_plot, classify_output_heatmap]
                )

        # Login & Register Button Logic
        login_btn.click(
            fn=login_user,
            inputs=[login_username, login_password],
            outputs=[search_section, classify_section, login_msg, user_id_state, username_state]
        )

        register_btn.click(
            fn=register_user,
            inputs=[login_username, login_password],
            outputs=[login_msg]
        )

    return demo


# Run app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(show_api=False)
