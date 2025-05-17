import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import requests
import json
import datetime
from utils.database import Database
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DB connection
db = Database()
db.create_tables()  # Ensure tables exist

# Base API URL (assuming app.py runs on this port)
API_URL = "http://localhost:8000"


# Helper functions for auth and API calls
def make_api_request(endpoint, method="GET", data=None, files=None, auth=None):
    """Helper to make API requests with proper error handling"""
    url = f"{API_URL}/{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, auth=auth)
        elif method == "POST":
            response = requests.post(url, data=data, files=files, auth=auth)

        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise gr.Error(f"API request failed: {str(e)}")


# Authentication functions
def register_user(username, password):
    """Register a new user in the database"""
    if not username or not password:
        return "❌ Username and password cannot be empty."

    try:
        db.register_user(username, password)
        return f"✅ User '{username}' registered successfully! Please login."
    except Exception as e:
        if 'unique constraint' in str(e).lower():
            return f"⚠️ Username '{username}' is already taken."
        return f"❌ Registration failed: {str(e)}"


def login_user(username, password):
    """Login a user and update UI visibility"""
    user_id, role = db.authenticate_user(username, password)
    if user_id is None:
        return (
            gr.update(visible=False),  # search section
            gr.update(visible=False),  # classify section
            gr.update(visible=False),  # feedback section
            "❌ Invalid credentials.",  # login message
            None,  # user_id state
            None  # username state
        )

    return (
        gr.update(visible=True),  # search section
        gr.update(visible=True),  # classify section
        gr.update(visible=True),  # feedback section
        f"✅ Welcome, {username}!",  # login message
        user_id,  # user_id state
        username  # username state
    )


# Feedback functionality
def submit_feedback(feedback_text, user_id, username):
    """Submit user feedback to the database"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to submit feedback.")

    if not feedback_text.strip():
        return "❌ Feedback cannot be empty."

    try:
        # Use direct DB call to save feedback
        db.save_feedback(user_id, feedback_text)
        return "✅ Thank you for your feedback! We appreciate your input."
    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}")
        return f"❌ Failed to submit feedback: {str(e)}"


def get_feedbacks(user_id, username):
    """Retrieve user's previous feedback"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to view your feedback.")

    try:
        feedbacks = db.get_feedbacks(user_id)
        if not feedbacks:
            return "You haven't submitted any feedback yet."

        formatted = []
        for feedback_text, created_at in feedbacks:
            # Format timestamp if it's a datetime object
            if isinstance(created_at, datetime.datetime):
                timestamp = created_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = str(created_at)

            formatted.append(f"[{timestamp}] {feedback_text}")

        return "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Failed to retrieve feedback: {str(e)}")
        return f"❌ Failed to retrieve feedback: {str(e)}"


# Recent activity functions
def load_recent_queries(user_id, username):
    """Load recent search queries for the current user"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to view recent queries.")

    try:
        # Make API request to get recent queries
        auth = (username, "password-placeholder")  # You'll need to handle this securely
        response = make_api_request("recent_queries", auth=auth)

        if not response.get("recent_queries"):
            return "No recent queries found."

        formatted = []
        for query in response["recent_queries"]:
            # Format depends on your API response structure
            query_text, image_path, username, timestamp = query
            formatted.append(f"{timestamp} | {query_text} | {image_path}")

        return "\n".join(formatted)
    except Exception as e:
        # Fallback to direct DB access if API fails
        try:
            rows = db.get_recent_queries(user_id=user_id)
            if not rows:
                return "No recent queries found."
            return "\n".join([f"{row[3]} | {row[0]} | {row[1]}" for row in rows])
        except Exception as db_e:
            logger.error(f"Failed to get recent queries: {str(e)}, DB fallback failed: {str(db_e)}")
            return f"❌ Failed to retrieve recent queries"


def get_top_queries():
    """Get the most frequently used search queries"""
    try:
        response = make_api_request("top-queries")
        top_queries = response.get("top_queries", [])

        if not top_queries:
            return "No popular queries found."

        formatted = []
        for item in top_queries:
            formatted.append(f"{item['query']}: {item['count']} searches")

        return "\n".join(formatted)
    except Exception as e:
        # Fallback to direct DB access
        try:
            queries = db.get_top_queries(limit=5)
            if not queries:
                return "No popular queries found."
            return "\n".join([f"{q}: {c} searches" for q, c in queries])
        except Exception as db_e:
            logger.error(f"Failed to get top queries: {str(e)}, DB fallback failed: {str(db_e)}")
            return "❌ Failed to retrieve popular queries"


# Image classification function
def classify_image(image, user_id, username, password):
    """Classify an image using the CLIP model API"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to classify images.")

    if image is None:
        raise gr.Error("🖼️ Please upload an image.")

    try:
        # Convert image to bytes for API request
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create file object for multipart/form-data request
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}

        # Make API request with Basic Auth
        auth = (username, password)
        response = make_api_request("classify_image", method="POST", files=files, auth=auth)

        top_probs = response.get("top_probs", [])
        top_classes = response.get("top_classes", [])

        # Format classification results
        results_text = "\n".join([f"{cls}: {prob * 100:.2f}%" for cls, prob in zip(top_classes, top_probs)])

        # Create chart
        fig = create_classification_plot(top_probs, top_classes)

        return results_text, fig
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        return f"❌ Classification failed: {str(e)}", None


def create_classification_plot(probs, class_names):
    """Create a horizontal bar chart for classification probabilities"""
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


# Image search function
def search_images(query, query_image, top_k, user_id, username, password):
    """Search for images using text query or image query"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to perform search.")

    if not query and query_image is None:
        raise gr.Error("Please provide either a text query or an image.")

    try:
        auth = (username, password)

        if query and not query_image:
            # Text-based search
            data = {'query': query, 'top_k': top_k}
            response = make_api_request("search_images", method="POST", data=data, auth=auth)
        else:
            # Image-based search
            img_byte_arr = io.BytesIO()
            query_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            files = {'file': ('query_image.jpg', img_byte_arr, 'image/jpeg')}
            data = {'top_k': top_k}
            response = make_api_request("search_images", method="POST", data=data, files=files, auth=auth)

        # Process search results
        results = response.get("results", [])
        if not results:
            return []

        images_with_labels = []
        for result in results:
            similarity = result["similarity"]
            image_base64 = result["image_base64"]

            # Decode base64 image
            image_data = base64.b64decode(image_base64.split(",")[1])
            pil_image = Image.open(io.BytesIO(image_data))

            label = f"Similarity: {similarity:.2f}"
            images_with_labels.append((pil_image, label))

        return images_with_labels
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise gr.Error(f"Search failed: {str(e)}")


# Dropdown suggestion function
def suggest_queries():
    """Get query suggestions from recent searches"""
    try:
        top_queries = db.get_top_queries(limit=5)
        return gr.Dropdown.update(choices=[q[0] for q in top_queries],
                                  visible=True) if top_queries else gr.Dropdown.update(visible=False)
    except Exception as e:
        logger.error(f"Failed to get query suggestions: {str(e)}")
        return gr.Dropdown.update(visible=False)


def fill_textbox(choice):
    """Fill textbox with selected suggestion"""
    return gr.Textbox.update(value=choice)


# Main Gradio interface
def create_interface():
    with gr.Blocks(title="CLIP Image Search & Classification") as demo:
        gr.Markdown("# 🖼️ CLIP Image Search & Classification with 🔐 Authentication")

        # Store user state
        user_id_state = gr.State(None)
        username_state = gr.State(None)
        password_state = gr.State(None)  # Store password for API calls

        # Authentication UI
        with gr.Row():
            login_username = gr.Textbox(label="Username", placeholder="Enter username")
            login_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")

        with gr.Row():
            login_btn = gr.Button("Login", variant="primary")
            register_btn = gr.Button("Register")

        login_msg = gr.Textbox(label="Status", interactive=False)

        # Main functionality (initially hidden)
        with gr.Tabs(visible=False) as main_tabs:
            # Search tab
            with gr.TabItem("Image Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(label="Text Query", placeholder="Search for images...")
                        dropdown = gr.Dropdown(choices=[], label="Suggestions", interactive=True, visible=False)
                        image_query = gr.Image(label="Or Upload Image Query", type="pil")
                        top_k_input = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Number of Results")
                        search_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=2):
                        results_gallery = gr.Gallery(label="Search Results", columns=[2], object_fit="contain",
                                                     allow_preview=True)

            # Classification tab
            with gr.TabItem("Image Classification"):
                with gr.Row():
                    with gr.Column(scale=1):
                        classify_input = gr.Image(label="Upload Image to Classify", type="pil")
                        classify_btn = gr.Button("Classify", variant="primary")

                    with gr.Column(scale=2):
                        with gr.Row():
                            classify_output_text = gr.Textbox(label="Top Classes", lines=6, interactive=False)
                            classify_output_plot = gr.Plot(label="Probability Distribution")

            # Recent Activity tab
            with gr.TabItem("Recent Activity"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔍 Recent Queries")
                        queries_btn = gr.Button("Show Recent Queries")
                        recent_queries_output = gr.Textbox(label="Your Recent Searches", lines=8, interactive=False)

                    with gr.Column():
                        gr.Markdown("### 📊 Popular Searches")
                        top_queries_btn = gr.Button("Show Popular Searches")
                        top_queries_output = gr.Textbox(label="Most Popular Searches", lines=8, interactive=False)

            # Feedback tab
            with gr.TabItem("Feedback"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 💬 Submit Feedback")
                        feedback_textbox = gr.Textbox(
                            label="Your Feedback",
                            lines=5,
                            placeholder="Tell us what you think about the application..."
                        )
                        feedback_btn = gr.Button("Submit Feedback", variant="primary")
                        feedback_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        gr.Markdown("### 📝 Your Previous Feedback")
                        view_feedback_btn = gr.Button("View My Feedback")
                        previous_feedback = gr.Textbox(label="Previous Submissions", lines=8, interactive=False)

        # Connect search functionality
        text_input.focus(fn=suggest_queries, inputs=[], outputs=[dropdown])
        dropdown.select(fn=fill_textbox, inputs=[dropdown], outputs=[text_input])
        dropdown.select(fn=lambda: gr.Dropdown.update(visible=False), inputs=[], outputs=[dropdown])

        search_btn.click(
            fn=search_images,
            inputs=[text_input, image_query, top_k_input, user_id_state, username_state, password_state],
            outputs=[results_gallery]
        )

        # Connect classification functionality
        classify_btn.click(
            fn=classify_image,
            inputs=[classify_input, user_id_state, username_state, password_state],
            outputs=[classify_output_text, classify_output_plot]
        )

        # Connect recent activity functionality
        queries_btn.click(
            fn=load_recent_queries,
            inputs=[user_id_state, username_state],
            outputs=[recent_queries_output]
        )

        top_queries_btn.click(
            fn=get_top_queries,
            inputs=[],
            outputs=[top_queries_output]
        )

        # Connect feedback functionality
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[feedback_textbox, user_id_state, username_state],
            outputs=[feedback_status]
        )

        view_feedback_btn.click(
            fn=get_feedbacks,
            inputs=[user_id_state, username_state],
            outputs=[previous_feedback]
        )

        # Authentication logic
        login_btn.click(
            fn=login_user,
            inputs=[login_username, login_password],
            outputs=[main_tabs, main_tabs, main_tabs, login_msg, user_id_state, username_state]
        ).then(
            fn=lambda pw: pw,  # Pass password to state
            inputs=[login_password],
            outputs=[password_state]
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