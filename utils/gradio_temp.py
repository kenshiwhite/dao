import gradio as gr
import requests
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import json
from typing import Optional, Tuple, List

# FastAPI server configuration
API_BASE_URL = "http://localhost:8000"  # Adjust this to your FastAPI server URL


class FastAPIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def set_auth(self, username: str, password: str):
        """Set basic authentication for the session"""
        self.session.auth = (username, password)

    def clear_auth(self):
        """Clear authentication"""
        self.session.auth = None

    def register_user(self, username: str, password: str, role: str = "user") -> dict:
        """Register a new user"""
        data = {
            "username": username,
            "password": password,
            "role": role
        }
        response = self.session.post(f"{self.base_url}/register", json=data)
        return response.json(), response.status_code

    def login_user(self, username: str, password: str) -> dict:
        """Login user and return user info"""
        data = {
            "username": username,
            "password": password
        }
        response = self.session.post(f"{self.base_url}/login", json=data)
        return response.json(), response.status_code

    def get_user_profile(self) -> dict:
        """Get current user's profile"""
        response = self.session.get(f"{self.base_url}/user/profile")
        return response.json(), response.status_code

    def classify_image(self, image: Image.Image) -> dict:
        """Classify an image"""
        # Convert PIL image to bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)

        files = {"file": ("image.jpg", img_buffer, "image/jpeg")}
        response = self.session.post(f"{self.base_url}/classify_image", files=files)
        return response.json(), response.status_code

    def search_images(self, query: Optional[str] = None, image: Optional[Image.Image] = None, top_k: int = 10) -> dict:
        """Search images by text query or image"""
        data = {"top_k": top_k}
        files = {}

        if query:
            data["query"] = query

        if image:
            img_buffer = BytesIO()
            image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            files["file"] = ("query_image.jpg", img_buffer, "image/jpeg")

        response = self.session.post(f"{self.base_url}/search_images", data=data, files=files)
        return response.json(), response.status_code

    def submit_feedback(self, feedback_text: str) -> dict:
        """Submit user feedback"""
        data = {"feedback_text": feedback_text}
        response = self.session.post(f"{self.base_url}/feedback", json=data)
        return response.json(), response.status_code

    def get_feedback(self) -> dict:
        """Get user's feedback"""
        response = self.session.get(f"{self.base_url}/feedback")
        return response.json(), response.status_code

    def delete_feedback(self, feedback_id: int) -> dict:
        """Delete feedback by ID"""
        response = self.session.delete(f"{self.base_url}/feedback/{feedback_id}")
        return response.json(), response.status_code

    def get_recent_queries(self) -> dict:
        """Get user's recent queries"""
        response = self.session.get(f"{self.base_url}/recent_queries")
        return response.json(), response.status_code

    def get_recent_classifications(self) -> dict:
        """Get user's recent classifications"""
        response = self.session.get(f"{self.base_url}/recent_classifications")
        return response.json(), response.status_code

    def get_top_queries(self, limit: int = 3) -> dict:
        """Get most searched queries"""
        response = self.session.get(f"{self.base_url}/top-queries", params={"limit": limit})
        return response.json(), response.status_code

    def admin_upload_image(self, image: Image.Image) -> dict:
        """Admin: Upload image to index"""
        img_buffer = BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)

        files = {"file": ("admin_image.jpg", img_buffer, "image/jpeg")}
        response = self.session.post(f"{self.base_url}/admin/upload_image", files=files)
        return response.json(), response.status_code


# Initialize the FastAPI client
api_client = FastAPIClient()


def register_user(username: str, password: str) -> str:
    """Register a new user"""
    if not username or not password:
        return "❌ Username and password cannot be empty."

    try:
        result, status_code = api_client.register_user(username, password)

        if status_code == 201:
            return f"✅ User '{username}' registered successfully!"
        else:
            error_msg = result.get("detail", "Registration failed")
            if "already exists" in error_msg:
                return f"⚠️ Username '{username}' is already taken."
            return f"❌ Registration failed: {error_msg}"
    except Exception as e:
        return f"❌ Registration failed: {str(e)}"


def login_user(username: str, password: str) -> Tuple:
    """Login user and update interface visibility"""
    if not username or not password:
        return (
            gr.update(visible=False),  # search_section
            gr.update(visible=False),  # classify_section
            gr.update(visible=False),  # admin_section
            gr.update(visible=False),  # feedback_section
            gr.update(visible=False),  # top_queries_section
            "❌ Username and password cannot be empty.",
            None,  # user_id_state
            None,  # username_state
            None  # user_role_state
        )

    try:
        # Set authentication for the client
        api_client.set_auth(username, password)

        # Try to get user profile to verify authentication
        profile_result, profile_status = api_client.get_user_profile()

        if profile_status == 200:
            user_role = profile_result.get("role", "user")
            user_id = profile_result.get("user_id")
            is_admin = user_role == "admin"

            return (
                gr.update(visible=True),  # search_section
                gr.update(visible=True),  # classify_section
                gr.update(visible=is_admin),  # admin_section
                gr.update(visible=True),  # feedback_section
                gr.update(visible=True),  # top_queries_section
                f"✅ Welcome, {username}! Role: {user_role}",
                user_id,  # user_id_state
                username,  # username_state
                user_role  # user_role_state
            )
        else:
            api_client.clear_auth()
            error_msg = profile_result.get("detail", "Login failed")
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                f"❌ {error_msg}",
                None,
                None,
                None
            )
    except Exception as e:
        api_client.clear_auth()
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            f"❌ Login failed: {str(e)}",
            None,
            None,
            None
        )


def classify_image(image: Image.Image, user_id: int, username: str) -> str:
    """Classify an uploaded image"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to classify images.")
    if image is None:
        raise gr.Error("🖼️ Please upload an image.")

    try:
        result, status_code = api_client.classify_image(image)

        if status_code == 200:
            top_classes = result["top_classes"]
            top_probs = result["top_probs"]
            results_text = "\n".join([f"{cls}: {prob * 100:.2f}%" for cls, prob in zip(top_classes, top_probs)])
            return results_text
        else:
            error_msg = result.get("detail", "Classification failed")
            raise gr.Error(f"❌ {error_msg}")
    except Exception as e:
        raise gr.Error(f"❌ Classification failed: {str(e)}")


def search_images(query: str, query_image: Image.Image, top_k: int, user_id: int, username: str) -> List:
    """Search images by text query or image"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to perform search.")
    if not query and query_image is None:
        raise gr.Error("Please provide either a text query or an image.")

    try:
        result, status_code = api_client.search_images(query=query if query else None,
                                                       image=query_image,
                                                       top_k=top_k)

        if status_code == 200:
            results = result["results"]
            images_with_labels = []

            for item in results:
                # Decode base64 image
                img_data = item["image_base64"].split(",")[1]  # Remove data:image/jpeg;base64, prefix
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes))

                label = f"Similarity: {item['similarity']:.3f}"
                images_with_labels.append((img, label))

            return images_with_labels
        else:
            error_msg = result.get("detail", "Search failed")
            raise gr.Error(f"❌ {error_msg}")
    except Exception as e:
        raise gr.Error(f"❌ Search failed: {str(e)}")


def submit_feedback(feedback_text: str, user_id: int, username: str) -> Tuple[str, str]:
    """Submit user feedback"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to submit feedback.")
    if not feedback_text:
        raise gr.Error("Please enter some feedback.")

    try:
        result, status_code = api_client.submit_feedback(feedback_text)

        if status_code == 200:
            return "✅ Thank you for your feedback!", ""
        else:
            error_msg = result.get("detail", "Failed to submit feedback")
            return f"❌ {error_msg}", feedback_text
    except Exception as e:
        return f"❌ Error submitting feedback: {str(e)}", feedback_text


def load_recent_feedback(user_id: int, username: str, user_role: str) -> str:
    """Load recent feedback"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to view feedback.")

    try:
        result, status_code = api_client.get_feedback()

        if status_code == 200:
            feedbacks = result.get("feedbacks", [])

            if not feedbacks:
                return "No feedback submitted yet."

            formatted_feedback = []
            is_admin = user_role == "admin"

            for feedback in feedbacks:
                if is_admin and len(feedback) >= 5:
                    # Admin view: [id, user_id, username, feedback_text, created_at]
                    feedback_id, f_user_id, f_username, feedback_text, created_at = feedback[:5]
                    formatted_feedback.append(
                        f"ID: {feedback_id} | User: {f_username} (ID: {f_user_id}) | {created_at} | {feedback_text}"
                    )
                else:
                    # User view: [id, feedback_text, created_at]
                    feedback_id, feedback_text, created_at = feedback[:3]
                    formatted_feedback.append(f"ID: {feedback_id} | {created_at} | {feedback_text}")

            return "\n".join(formatted_feedback)
        else:
            error_msg = result.get("detail", "Failed to load feedback")
            return f"❌ {error_msg}"
    except Exception as e:
        return f"❌ Error loading feedback: {str(e)}"


def delete_user_feedback(feedback_id: int, user_id: int, username: str, user_role: str) -> Tuple[str, str]:
    """Delete user feedback"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to delete feedback.")

    if not feedback_id:
        return "❌ Please enter a valid feedback ID", None

    try:
        result, status_code = api_client.delete_feedback(int(feedback_id))

        if status_code == 200:
            updated_feedback = load_recent_feedback(user_id, username, user_role)
            return "✅ Feedback deleted successfully!", updated_feedback
        else:
            error_msg = result.get("detail", "Failed to delete feedback")
            return f"❌ {error_msg}", None
    except Exception as e:
        return f"❌ Error deleting feedback: {str(e)}", None


def load_recent_queries(user_id: int, username: str) -> str:
    """Load recent queries"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to view recent queries.")

    try:
        result, status_code = api_client.get_recent_queries()

        if status_code == 200:
            queries = result.get("recent_queries", [])
            if not queries:
                return "No recent queries."

            # Format: [query_text, image_path, created_at]
            formatted_queries = []
            for query in queries:
                if len(query) >= 3:
                    query_text, image_path, created_at = query[:3]
                    formatted_queries.append(f"{created_at} | {query_text} | {image_path}")

            return "\n".join(formatted_queries) if formatted_queries else "No recent queries."
        else:
            error_msg = result.get("detail", "Failed to load queries")
            return f"❌ {error_msg}"
    except Exception as e:
        return f"❌ Error loading queries: {str(e)}"


def load_recent_classifications(user_id: int, username: str) -> str:
    """Load recent classifications"""
    if user_id is None:
        raise gr.Error("🔒 Please log in to view recent classifications.")

    try:
        result, status_code = api_client.get_recent_classifications()

        if status_code == 200:
            classifications = result.get("recent_classifications", [])
            if not classifications:
                return "No recent classifications."

            formatted_classifications = []
            for classification in classifications:
                # Format depends on your database schema
                formatted_classifications.append(str(classification))

            return "\n".join(formatted_classifications)
        else:
            error_msg = result.get("detail", "Failed to load classifications")
            return f"❌ {error_msg}"
    except Exception as e:
        return f"❌ Error loading classifications: {str(e)}"


def load_top_queries() -> str:
    """Load top searched queries"""
    try:
        result, status_code = api_client.get_top_queries(limit=3)

        if status_code == 200:
            top_queries = result.get("top_queries", [])
            if not top_queries:
                return "No search queries found."

            formatted_queries = "\n".join([
                f"'{item['query']}' - {item['count']} searches"
                for item in top_queries
            ])
            return formatted_queries
        else:
            return "Error loading top queries."
    except Exception as e:
        return f"Error loading top queries: {str(e)}"


def admin_upload_image(image: Image.Image, user_id: int, username: str, user_role: str) -> str:
    """Admin: Upload image to index"""
    if user_id is None:
        raise gr.Error("🔒 Please log in.")
    if user_role != "admin":
        raise gr.Error("🔐 Admin privileges required.")
    if image is None:
        raise gr.Error("Please upload an image.")

    try:
        result, status_code = api_client.admin_upload_image(image)

        if status_code == 201:
            return "✅ Image added to index and saved."
        else:
            error_msg = result.get("detail", "Upload failed")
            return f"❌ {error_msg}"
    except Exception as e:
        return f"❌ Upload failed: {str(e)}"


def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="CLIP Auth Image Search") as demo:
        gr.Markdown("# 🖼️ CLIP Image Search & Classification with 🔐 Authentication")

        # Login section
        with gr.Row():
            with gr.Column():
                login_username = gr.Textbox(label="Username")
                login_password = gr.Textbox(label="Password", type="password")
                with gr.Row():
                    login_btn = gr.Button("Login", variant="primary")
                    register_btn = gr.Button("Register")
                login_msg = gr.Textbox(label="Status", interactive=False)

        # State variables
        user_id_state = gr.State()
        username_state = gr.State()
        user_role_state = gr.State()

        # Top queries section
        with gr.Row(visible=False) as top_queries_section:
            with gr.Column():
                gr.Markdown("## 🔍 Top Searched Queries")
                top_queries_output = gr.Textbox(label="Most Popular Searches", lines=4, interactive=False)
                refresh_top_queries_btn = gr.Button("Refresh Top Queries")

        # Search section
        with gr.Row(visible=False) as search_section:
            with gr.Tab("🔍 Search"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(label="Text Query", placeholder="Enter search query...")
                        image_query = gr.Image(label="Image Query", type="pil")
                        top_k_input = gr.Number(value=4, label="Top K Results", minimum=1, maximum=20)
                        search_btn = gr.Button("Search", variant="primary")
                    with gr.Column():
                        results_gallery = gr.Gallery(
                            label="Search Results",
                            columns=2,
                            object_fit="contain",
                            allow_preview=True
                        )

            with gr.Tab("📊 Recent Activity"):
                with gr.Row():
                    queries_btn = gr.Button("🔍 Show Recent Queries")
                    classifications_btn = gr.Button("📊 Show Recent Classifications")

                recent_queries_output = gr.Textbox(label="Recent Queries", lines=8, interactive=False)
                recent_classifications_output = gr.Textbox(label="Recent Classifications", lines=8, interactive=False)

        # Classification section
        with gr.Row(visible=False) as classify_section:
            with gr.Tab("🎯 Classification"):
                with gr.Row():
                    with gr.Column():
                        classify_input = gr.Image(label="Upload Image", type="pil")
                        classify_btn = gr.Button("Classify", variant="primary")
                    with gr.Column():
                        classify_output_text = gr.Textbox(
                            label="Top Class Probabilities",
                            lines=8,
                            interactive=False
                        )

        # Admin section
        with gr.Row(visible=False) as admin_section:
            with gr.Tab("⚙️ Admin Upload"):
                with gr.Row():
                    with gr.Column():
                        admin_image = gr.Image(label="Upload Image", type="pil")
                        upload_btn = gr.Button("Add to Index", variant="primary")
                    with gr.Column():
                        upload_status = gr.Textbox(label="Upload Result", interactive=False)

        # Feedback section
        with gr.Row(visible=False) as feedback_section:
            with gr.Tab("💬 Feedback"):
                with gr.Row():
                    with gr.Column(scale=2):
                        feedback_text = gr.Textbox(
                            label="Share Your Feedback",
                            placeholder="Tell us what you think...",
                            lines=4
                        )
                        feedback_btn = gr.Button("Submit Feedback", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### Feedback Management
                        - Regular users can delete their own feedback
                        - Admins can delete any feedback
                        """)

                feedback_status = gr.Markdown()

                gr.Markdown("### View and Manage Feedback")
                with gr.Row():
                    view_feedback_btn = gr.Button("🔄 View Feedback History")

                feedback_history = gr.TextArea(
                    label="Feedback History",
                    lines=8,
                    interactive=False
                )

                with gr.Row():
                    with gr.Column():
                        feedback_id_input = gr.Number(
                            label="Feedback ID to Delete",
                            precision=0,
                            minimum=1
                        )
                    with gr.Column():
                        delete_feedback_btn = gr.Button("🗑️ Delete Feedback", variant="stop")

        # Event handlers
        login_btn.click(
            fn=login_user,
            inputs=[login_username, login_password],
            outputs=[
                search_section, classify_section, admin_section,
                feedback_section, top_queries_section, login_msg,
                user_id_state, username_state, user_role_state
            ]
        )

        register_btn.click(
            fn=register_user,
            inputs=[login_username, login_password],
            outputs=[login_msg]
        )

        search_btn.click(
            fn=search_images,
            inputs=[text_input, image_query, top_k_input, user_id_state, username_state],
            outputs=[results_gallery]
        )

        classify_btn.click(
            fn=classify_image,
            inputs=[classify_input, user_id_state, username_state],
            outputs=[classify_output_text]
        )

        upload_btn.click(
            fn=admin_upload_image,
            inputs=[admin_image, user_id_state, username_state, user_role_state],
            outputs=[upload_status]
        )

        feedback_btn.click(
            fn=submit_feedback,
            inputs=[feedback_text, user_id_state, username_state],
            outputs=[feedback_status, feedback_text]
        )

        view_feedback_btn.click(
            fn=load_recent_feedback,
            inputs=[user_id_state, username_state, user_role_state],
            outputs=[feedback_history]
        )

        delete_feedback_btn.click(
            fn=delete_user_feedback,
            inputs=[feedback_id_input, user_id_state, username_state, user_role_state],
            outputs=[feedback_status, feedback_history]
        )

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

        refresh_top_queries_btn.click(
            fn=load_top_queries,
            inputs=[],
            outputs=[top_queries_output]
        )

        # Clear feedback ID after deletion
        delete_feedback_btn.click(
            fn=lambda: gr.Number.update(value=None),
            inputs=[],
            outputs=[feedback_id_input]
        )

        # Load top queries on startup
        demo.load(
            fn=load_top_queries,
            inputs=None,
            outputs=[top_queries_output]
        )

    return demo


if __name__ == "__main__":
    # Make sure your FastAPI server is running on localhost:8000
    print("Starting Gradio interface...")
    print("Make sure your FastAPI server is running on http://localhost:8000")

    demo = create_interface()
    demo.launch(
        show_api=False,
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Gradio default port
        share=False  # Set to True if you want a public link
    )