from app.app import create_interface
from scripts.preprocess_data import preprocess_dataset
import os


def main():
    # Check if precomputed data exists
    if not (os.path.exists("data/saved_features.pt") and
            os.path.exists("data/saved_images.pt")):
        print("Preprocessing data...")
        preprocess_dataset()

    # Launch the app
    app = create_interface()
    app.launch()


if __name__ == "__main__":
    main()