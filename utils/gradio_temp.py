import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from app.app import CLIPBackend
from PIL import Image
import asyncio
from utils.database import Database

# Создание компонента dropdown
dropdown = gr.Dropdown(choices=[], label="Recent Queries", visible=False)

# Создание экземпляров backend и базы данных
backend = CLIPBackend()
db = Database()
db.create_tables()

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

    # Получаем результаты классификации
    probs, classes = backend.classify_image(image, class_names)
    plot = create_classification_plot(probs, classes)
    similarity_map = backend.get_similarity_map(image)
    heatmap = backend.generate_heatmap(image, similarity_map)

    # Сохраняем запрос в базе данных
    db.save_query("Image Classification", "image_classification.png")
    return plot, heatmap

def search_images_wrapper(query=None, query_image=None, top_k=4):
    return asyncio.run(search_images(query=query, query_image=query_image, top_k=top_k))

async def search_images(query=None, query_image=None, top_k=4):
    if not query and query_image is None:
        raise gr.Error("Please provide either a text query or an image.")

    # Получаем результаты поиска изображений
    results = await backend.search_images(query=query, query_image=query_image, top_k=top_k)

    images_with_labels = []
    heatmaps = []

    # Обрабатываем результаты поиска
    for img_tensor, score in results:
        pil_img = backend.tensor_to_pil(img_tensor)
        label = f"Similarity: {score:.2f}"

        similarity_map = backend.get_similarity_map(pil_img)
        heatmap = backend.generate_heatmap(pil_img, similarity_map)

        images_with_labels.append((pil_img, label))
        heatmaps.append(heatmap)

    # Сохраняем запрос в базе данных
    db.save_query(query if query else "Image Search", "search_image.png")
    return images_with_labels, heatmaps

def suggest_queries():
    # Получаем последние 5 запросов из базы данных
    recent_queries = db.get_recent_queries(limit=5)
    # Если запросы есть, обновляем dropdown
    if recent_queries:
        return dropdown.update(choices=recent_queries, visible=True)
    else:
        return dropdown.update(visible=False)

def add_query_to_database(query):
    # Сохраняем новый запрос в базе данных
    if query:
        db.save_query(query, "search_query.png")
    return suggest_queries()

def fill_textbox(choice):
    return gr.Textbox.update(value=choice)

def create_interface():
    with gr.Blocks(title="CLIP Image Search and Classification with Heatmap") as demo:
        gr.Markdown("# 🖼️ CLIP Image Search and Classification with Heatmap")

        with gr.Tab("Search"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Text Query", placeholder="Enter your query...")
                    dropdown = gr.Dropdown(choices=[], label="Suggestions", interactive=True, visible=False)
                    image_query = gr.Image(label="Image Query", type="pil")
                    search_btn = gr.Button("Search")
                with gr.Column():
                    results_gallery = gr.Gallery(label="Search Results", columns=[2], object_fit="contain", allow_preview=True)
                    heatmap_gallery = gr.Gallery(label="Heatmaps", columns=[2], object_fit="contain", allow_preview=True)

            # Показать dropdown при фокусе на textbox
            text_input.focus(fn=suggest_queries, inputs=[], outputs=[dropdown])

            # Подставить выбранное значение из выпадающего списка в поле ввода
            dropdown.change(fn=fill_textbox, inputs=[dropdown], outputs=[text_input])
            dropdown.change(fn=lambda: gr.Dropdown.update(visible=False), inputs=[], outputs=[dropdown])

            # После ввода текста сохраняем запрос и обновляем список
            text_input.submit(fn=add_query_to_database, inputs=[text_input], outputs=[dropdown])

            # Кнопка поиска, запускающая обработку запросов
            search_btn.click(
                fn=search_images_wrapper,
                inputs=[text_input, image_query],
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

            # Кнопка классификации, запускающая обработку изображений
            classify_btn.click(
                fn=classify_image,
                inputs=[classify_input],
                outputs=[classify_output_plot, classify_output_heatmap]
            )

    return demo

if __name__ == "__main__":
    # Запуск интерфейса
    demo = create_interface()
    demo.launch(show_api=False)
