import matplotlib.pyplot as plt
import numpy as np

def create_bar_chart(probabilities, labels, title="Zero-Shot Classification Results"):
    """Create a bar chart for classification results."""
    plt.figure(figsize=(10, 6))
    y = np.arange(len(probabilities))
    plt.barh(y, probabilities, color='skyblue')
    plt.yticks(y, labels)
    plt.xlabel("Probability")
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest probability at the top
    plt.tight_layout()
    return plt
