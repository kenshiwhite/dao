from utils.gradio_temp import  create_interface
from scripts.preprocess_data import preprocess_dataset
from utils.database import Database
import os
import atexit

def main():
    if not (os.path.exists("scripts/data/saved_features.pt") and
            os.path.exists("scripts/data/saved_images.pt")):
        print("Preprocessing data...")
        preprocess_dataset()

    db = Database()
    db.create_tables()
    atexit.register(db.close_all)

    app = create_interface()

    if app is not None:
        app.launch()
    else:
        print("Ошибка: интерфейс не создан.")

if __name__ == "__main__":
    main()
