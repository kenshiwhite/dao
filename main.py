from utils.gradio_temp import  create_interface
from scripts.preprocess_data import preprocess_dataset
from utils.database import Database
import os
import atexit

def main():
    # 1. Предобработка данных (если нет сохранённых файлов)
    if not (os.path.exists("scripts/data/saved_features.pt") and
            os.path.exists("scripts/data/saved_images.pt")):
        print("Preprocessing data...")
        preprocess_dataset()

    # 2. Инициализация базы данных
    db = Database()
    db.create_tables()  # Создание таблиц в базе данных
    atexit.register(db.close_all)  # Регистрация закрытия соединений при завершении работы

    # 3. Создание интерфейса
    app = create_interface()  # Использование функции create_interface из app.py

    if app is not None:
        app.launch()  # Запуск интерфейса
    else:
        print("Ошибка: интерфейс не создан.")

if __name__ == "__main__":
    main()
