import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
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

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(),
    title="Rubik's Cube Recognition",
    description="Upload an image of a Rubik's cube to detect its faces and colors.",
    examples=[
        ["examples/cube1.jpg"],
        ["examples/cube2.jpg"],
        ["examples/cube3.jpg"]
    ]
)

# Launch the interface
if __name__ == "__main__":
    iface.launch() 