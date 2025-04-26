import os
from utils.gradio_temp import create_interface
from scripts.preprocess_data import preprocess_dataset
from utils.database import database

def main():
    # Создание таблиц в БД
    try:
        db = database()
        db.create_tables()
        print("✅ Таблицы БД проверены/созданы.")
    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}")
        return

    # Проверка и предобработка данных
    if not (os.path.exists("scripts/data/saved_features.pt") and
            os.path.exists("scripts/data/saved_images.pt")):
        print("🔄 Предобработка данных...")
        preprocess_dataset()

    interface = create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
