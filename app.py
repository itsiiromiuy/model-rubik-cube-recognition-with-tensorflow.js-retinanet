import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from src.model.visualize import process_image
from src.data.shared import category_index


def predict(image):
    """
    Process the input image and return the visualization with detected objects
    """
    # Load the model
    model = tf.saved_model.load('exported_model')
    model_fn = model.signatures['serving_default']

    # Process image
    processed_image = process_image(image)

    # Get predictions
    result = model_fn(processed_image)

    # Visualize results
    visualization = visualize_detection(
        processed_image[0].numpy(),
        result['detection_boxes'][0].numpy(),
        result['detection_classes'][0].numpy().astype(int),
        result['detection_scores'][0].numpy(),
        category_index
    )

    return Image.fromarray(visualization)


def process_image(image):
    # Convert to RGB if needed
    if image is not None:
        image = Image.fromarray(image)
        # Add your model prediction logic here
        return "Rubik's cube detected! (Demo version)"
    return "No image provided"


# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs="text",
    title="Rubik's Cube Recognition",
    description="Upload an image of a Rubik's cube to detect and analyze it.",
    examples=[],
    theme=gr.themes.Soft()
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()
