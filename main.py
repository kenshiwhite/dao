import os
from app.app import demo  # Import the Gradio app
from scripts.preprocess_data import preprocess_data  # Import the preprocessing script

def main():
    # Check if precomputed features and images exist
    features_path = "data/saved_features.pt"
    images_path = "data/saved_images.pt"

    if not (os.path.exists(features_path) and os.path.exists(images_path)):
        print("Precomputed features and images not found. Running preprocessing script...")
        preprocess_data()  # Preprocess the data if files don't exist

    # Launch the Gradio app
    print("Launching the Gradio app...")
    demo.launch()

if __name__ == "__main__":
    main()