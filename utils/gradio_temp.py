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

def load_recent_queries(user_id, username):
    if user_id is None:
        raise gr.Error("🔒 Please log in to view recent queries.")
    rows = db.get_recent_queries(user_id=user_id)
    return "\n".join([f"{row[3]} | {row[0]} | {row[1]}" for row in rows]) if rows else "No recent queries."

def load_recent_classifications(user_id, username):
    if user_id is None:
        raise gr.Error("🔒 Please log in to view recent classifications.")
    rows = db.get_recent_classifications(user_id)
    if not rows:
        return "No recent classifications."
    formatted = []
    for r in rows:
        formatted.append(
            f"{r['timestamp']} | {r['image_path']} | Classes: {', '.join(r['top_classes'])} | Probs: {', '.join(map(str, r['top_probs']))}"
        )
    return "\n".join(formatted)

def load_top_queries():
    rows = db.get_top_queries(limit=3)
    if not rows:
        return "No popular queries."
    return "\n".join([f"{query} — {count} times" for query, count in rows])


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
    similarity_map = backend.get_similarity_map(image)
    results_text = "\n".join([f"{cls}: {prob * 100:.2f}%" for cls, prob in zip(classes, probs)])
    db.save_query("Image Classification", "image_classification.png", user_id)
    return results_text

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
    for img_tensor, score in results:
        pil_img = backend.tensor_to_pil(img_tensor)
        label = f"Similarity: {score:.2f}"
        similarity_map = backend.get_similarity_map(pil_img)
        images_with_labels.append((pil_img, label))
    db.save_query(query if query else "Image Search", "search_image.png", user_id)
    return images_with_labels

# Suggest dropdown
def suggest_queries():
    recent = db.get_recent_queries(limit=5)
    return gr.Dropdown.update(choices=[q[0] for q in recent], visible=True) if recent else gr.Dropdown.update(visible=False)

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
                        results_gallery = gr.Gallery(label="Search Results", columns=[2], object_fit="contain", allow_preview=True)

                text_input.focus(fn=suggest_queries, inputs=[], outputs=[dropdown])
                dropdown.change(fn=fill_textbox, inputs=[dropdown], outputs=[text_input])
                dropdown.change(fn=lambda: gr.Dropdown.update(visible=False), inputs=[], outputs=[dropdown])

                search_btn.click(
                    fn=search_images_wrapper,
                    inputs=[text_input, image_query, top_k_input, user_id_state, username_state],
                    outputs=[results_gallery]
                )

                with gr.Tab("Recent Activity"):
                    with gr.Row():
                        queries_btn = gr.Button("🔍 Show Recent Queries")
                        classifications_btn = gr.Button("📊 Show Recent Classifications")
                        top_queries_btn = gr.Button("🔥 Show Most Popular Queries")
                    with gr.Row():
                        recent_queries_output = gr.Textbox(label="Recent Queries", lines=8, interactive=False)
                        recent_classifications_output = gr.Textbox(label="Recent Classifications", lines=8, interactive=False)
                        top_queries_output = gr.Textbox(label="Top Queries", lines=4, interactive=False)

                    queries_btn.click(
                        fn=load_recent_queries,
                        inputs=[user_id_state, username_state],
                        outputs=[recent_queries_output]
                    )
                    classifications_btn.click(
                        fn=load_recent_classifications,
                        inputs=[user_id_state, username_state],
                        outputs=[recent_classifications_output]
                    )
                    top_queries_btn.click(
                        fn=load_top_queries,
                        inputs=[],
                        outputs=[top_queries_output]
                    )

        with gr.Row(visible=False) as classify_section:
            with gr.Tab("Classification"):
                with gr.Row():
                    with gr.Column():
                        classify_input = gr.Image(label="Upload Image", type="pil")
                        classify_btn = gr.Button("Classify")
                    with gr.Column():
                        classify_output_text = gr.Textbox(label="Top Class Probabilities", lines=6, interactive=False)

                classify_btn.click(
                    fn=classify_image,
                    inputs=[classify_input, user_id_state, username_state],
                    outputs=[classify_output_text]
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
