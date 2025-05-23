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
    return "\n".join([f"{row[2]} | {row[0]} | {row[1]}" for row in rows]) if rows else "No recent queries."


def delete_user_feedback(feedback_id, user_id, username, password):
    if user_id is None:
        raise gr.Error("🔒 Please log in to delete feedback.")

    if not feedback_id:
        return "❌ Please enter a valid feedback ID", None

    try:
        # Re-authenticate to confirm user credentials
        user_id, role = db.authenticate_user(username, password)
        is_admin = role == "admin"

        success = db.delete_feedback(feedback_id, user_id, is_admin)

        if success:
            return "✅ Feedback deleted successfully!", load_recent_feedback(user_id, username)
        else:
            if is_admin:
                return "❌ Feedback not found!", None
            else:
                return "❌ You can only delete your own feedback!", None

    except Exception as e:
        return f"❌ Error deleting feedback: {str(e)}", None


def load_recent_feedback(user_id, username):
    if user_id is None:
        raise gr.Error("🔒 Please log in to view feedback.")

    # Get user role to determine what feedback to show
    _, role = db.authenticate_user_by_id(user_id)
    is_admin = role == "admin"

    # Get all feedback if admin, or just user's feedback otherwise
    if is_admin:
        feedbacks = db.get_feedbacks()  # Get all feedback
    else:
        feedbacks = db.get_feedbacks(user_id=user_id)  # Get only user's feedback

    if not feedbacks:
        return "No feedback submitted yet."

    # Format feedback with IDs for display
    formatted_feedback = []

    if is_admin:
        # For admin, show user_id and username
        for feedback in feedbacks:
            feedback_id = feedback[0]
            user_id = feedback[1]
            username = feedback[2]
            feedback_text = feedback[3]
            created_at = feedback[4]

            formatted_feedback.append(
                f"ID: {feedback_id} | User: {username} (ID: {user_id}) | {created_at} | {feedback_text}")
    else:
        # For regular users
        for feedback in feedbacks:
            feedback_id = feedback[0]
            feedback_text = feedback[1]
            created_at = feedback[2]

            formatted_feedback.append(f"ID: {feedback_id} | {created_at} | {feedback_text}")

    return "\n".join(formatted_feedback)

def load_top_queries():
    # Get the top 3 most searched queries from database
    top_queries = db.get_top_queries(limit=3)
    if not top_queries:
        return "No search queries found."

    formatted_queries = "\n".join([f"'{query}' - {count} searches" for query, count in top_queries])
    return formatted_queries


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


def login_user(username, password):
    user_id, role = db.authenticate_user(username, password)
    if user_id is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),  # top_queries_section
            "❌ Invalid credentials.",
            None,
            None,
            None
        )

    is_admin = role == "admin"
    return (
        gr.update(visible=True),  # search_section
        gr.update(visible=True),  # classify_section
        gr.update(visible=is_admin),  # admin_section
        gr.update(visible=True),  # feedback_section
        gr.update(visible=True),  # top_queries_section
        f"✅ Welcome, {username}!",
        user_id,
        username,
        password  # now returning password_state
    )


def classify_image(image, user_id, username, class_names=None):
    if user_id is None:
        raise gr.Error("🔒 Please log in to classify images.")
    if image is None:
        raise gr.Error("🖼️ Please upload an image.")

    probs, classes = backend.classify_image(image, class_names)
    results_text = "\n".join([f"{cls}: {prob * 100:.2f}%" for cls, prob in zip(classes, probs)])

    db.save_query("Image Classification", "image_classification.png", user_id)
    return results_text


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
        images_with_labels.append((pil_img, label))

    db.save_query(query if query else "Image Search", "search_image.png", user_id)
    return images_with_labels


def submit_feedback(feedback_text, user_id, username):
    if user_id is None:
        raise gr.Error("🔒 Please log in to submit feedback.")
    if not feedback_text:
        raise gr.Error("Please enter some feedback.")

    try:
        db.save_feedback(user_id=user_id, feedback_text=feedback_text, user_name=username)
        return "✅ Thank you for your feedback!", ""
    except Exception as e:
        return f"❌ Error saving feedback: {str(e)}", feedback_text


def suggest_queries():
    recent = db.get_recent_queries(limit=5)
    return gr.Dropdown.update(choices=[q[0] for q in recent], visible=True) if recent else gr.Dropdown.update(
        visible=False)


def fill_textbox(choice):
    return gr.Textbox.update(value=choice)


def create_interface():
    with gr.Blocks(title="CLIP Auth Image Search") as demo:
        gr.Markdown("# 🖼️ CLIP Image Search & Classification with 🔐 Authentication")

        login_username = gr.Textbox(label="Username")
        login_password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        register_btn = gr.Button("Register")
        login_msg = gr.Textbox(label="Status", interactive=False)

        # States
        user_id_state = gr.State()
        username_state = gr.State()
        password_state = gr.State()

        # Top queries section - Visible once logged in
        with gr.Row(visible=False) as top_queries_section:
            # Replace gr.Box() with gr.Column()
            with gr.Column():
                gr.Markdown("## 🔍 Top Searched Queries")
                top_queries_output = gr.Textbox(label="Most Popular Searches", lines=4, interactive=False)
                refresh_top_queries_btn = gr.Button("Refresh Top Queries")

                # Load top queries when button is clicked
                refresh_top_queries_btn.click(
                    fn=load_top_queries,
                    inputs=[],
                    outputs=[top_queries_output]
                )

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

                text_input.focus(fn=suggest_queries, inputs=[], outputs=[dropdown])
                dropdown.change(fn=fill_textbox, inputs=[dropdown], outputs=[text_input])
                dropdown.change(fn=lambda: gr.Dropdown.update(visible=False), inputs=[], outputs=[dropdown])

                search_btn.click(
                    fn=search_images_wrapper,
                    inputs=[text_input, image_query, top_k_input, user_id_state, username_state],
                    outputs=[results_gallery]
                )

                with gr.Tab("Recent Activity"):
                    queries_btn = gr.Button("🔍 Show Recent Queries")
                    classifications_btn = gr.Button("📊 Show Recent Classifications")
                    recent_queries_output = gr.Textbox(label="Recent Queries", lines=8, interactive=False)
                    recent_classifications_output = gr.Textbox(label="Recent Classifications", lines=8,
                                                               interactive=False)

                    queries_btn.click(fn=load_recent_queries, inputs=[user_id_state, username_state],
                                      outputs=[recent_queries_output])

        with gr.Row(visible=False) as classify_section:
            with gr.Tab("Classification"):
                classify_input = gr.Image(label="Upload Image", type="pil")
                classify_btn = gr.Button("Classify")
                classify_output_text = gr.Textbox(label="Top Class Probabilities", lines=6, interactive=False)

                classify_btn.click(
                    fn=classify_image,
                    inputs=[classify_input, user_id_state, username_state],
                    outputs=[classify_output_text]
                )

        with gr.Row(visible=False) as admin_section:
            with gr.Tab("Admin Upload"):
                admin_image = gr.Image(label="Upload Image", type="pil")
                upload_btn = gr.Button("Add to Index")
                upload_status = gr.Textbox(label="Upload Result", interactive=False)

                def upload_to_backend(image, user_id, username, password):
                    if image is None:
                        raise gr.Error("Please upload an image.")
                    if user_id is None:
                        raise gr.Error("Please log in.")
                    _, role = db.authenticate_user(username, password)
                    if role != "admin":
                        raise gr.Error("🔐 Admin privileges required.")
                    backend.add_image_to_index(image)
                    return "✅ Image added to index and saved."

                upload_btn.click(
                    fn=upload_to_backend,
                    inputs=[admin_image, user_id_state, username_state, password_state],
                    outputs=[upload_status]
                )

        # Feedback Section
        with gr.Row(visible=False) as feedback_section:
            with gr.Tab("Feedback"):
                with gr.Row():
                    with gr.Column(scale=2):
                        feedback_text = gr.Textbox(
                            label="Share Your Feedback",
                            placeholder="Tell us what you think...",
                            lines=4
                        )
                        feedback_btn = gr.Button("Submit Feedback", variant="primary")

                    with gr.Column(scale=1):
                        # Add a note about feedback deletion
                        gr.Markdown("""
                           ### Feedback Management
                           - Regular users can delete their own feedback
                           - Admins can delete any feedback
                           """)

                with gr.Row():
                    feedback_status = gr.Markdown()

                gr.Markdown("### View and Manage Feedback")
                with gr.Row():
                    view_feedback_btn = gr.Button("🔄 View Feedback History")
                    # Clear column for spacing
                    with gr.Column():
                        pass

                with gr.Row():
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

                # Connect the buttons
                feedback_btn.click(
                    fn=submit_feedback,
                    inputs=[feedback_text, user_id_state, username_state],
                    outputs=[feedback_status, feedback_text]
                )

                view_feedback_btn.click(
                    fn=load_recent_feedback,
                    inputs=[user_id_state, username_state],
                    outputs=[feedback_history]
                )

                delete_feedback_btn.click(
                    fn=delete_user_feedback,
                    inputs=[feedback_id_input, user_id_state, username_state, password_state],
                    outputs=[feedback_status, feedback_history]
                )

                # Clear the feedback ID field after deletion
                delete_feedback_btn.click(
                    fn=lambda: gr.Number.update(value=None),
                    inputs=[],
                    outputs=[feedback_id_input]
                )

        login_btn.click(
            fn=login_user,
            inputs=[login_username, login_password],
            outputs=[
                search_section,
                classify_section,
                admin_section,
                feedback_section,
                top_queries_section,
                login_msg,
                user_id_state,
                username_state,
                password_state
            ]
        )

        register_btn.click(
            fn=register_user,
            inputs=[login_username, login_password],
            outputs=[login_msg]
        )

        # Load top queries automatically when the demo is loaded and user is logged in
        demo.load(
            fn=load_top_queries,
            inputs=None,
            outputs=[top_queries_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(show_api=False)