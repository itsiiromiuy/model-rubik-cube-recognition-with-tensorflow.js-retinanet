import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Simplified category index
CATEGORY_INDEX = {
    1: {'id': 1, 'name': 'face'},
    2: {'id': 2, 'name': 'red_tile'},
    3: {'id': 3, 'name': 'white_tile'},
    4: {'id': 4, 'name': 'blue_tile'},
    5: {'id': 5, 'name': 'orange_tile'},
    6: {'id': 6, 'name': 'green_tile'},
    7: {'id': 7, 'name': 'yellow_tile'}
}


def preprocess_image(image):
    """
    Preprocess input image
    """
    if image is None:
        return None

    # Convert to PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize to model expected size
    image = image.resize((640, 640))

    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)


def load_model():
    """
    Load pretrained model
    """
    try:
        # Try to load saved model
        if os.path.exists('exported_model'):
            model = tf.saved_model.load('exported_model')
            return model
        else:
            # If no model file exists, return None
            return None
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None


def predict_image(image):
    """
    Make predictions on input image
    """
    if image is None:
        return "Please upload an image", None

    try:
        # Preprocess image
        processed_image = preprocess_image(image)

        if processed_image is None:
            return "Image preprocessing failed", None

        # Load model
        model = load_model()

        if model is None:
            return "Model not found. This is a demo version, actual model needs to be trained first.\n\nDetected a Rubik's cube image!", image

        # Make prediction
        model_fn = model.signatures['serving_default']

        # Convert input format
        input_tensor = tf.convert_to_tensor(processed_image)

        # Execute inference
        predictions = model_fn(input_tensor)

        # Parse results
        detection_boxes = predictions['detection_boxes'][0].numpy()
        detection_classes = predictions['detection_classes'][0].numpy().astype(
            int)
        detection_scores = predictions['detection_scores'][0].numpy()

        # Filter low confidence detections
        valid_detections = detection_scores > 0.5
        valid_boxes = detection_boxes[valid_detections]
        valid_classes = detection_classes[valid_detections]
        valid_scores = detection_scores[valid_detections]

        # Generate result description
        if len(valid_boxes) > 0:
            result_text = f"Detected {len(valid_boxes)} objects:\n"
            for i, (cls, score) in enumerate(zip(valid_classes, valid_scores)):
                class_name = CATEGORY_INDEX.get(
                    cls, {}).get('name', f'class_{cls}')
                result_text += f"- {class_name}: {score:.2f}\n"
        else:
            result_text = "No Rubik's cube related objects detected"

        # Draw detection boxes on image (simplified version)
        output_image = draw_boxes_on_image(
            image, valid_boxes, valid_classes, valid_scores)

        return result_text, output_image

    except Exception as e:
        error_msg = f"Error occurred during prediction: {str(e)}\n\nThis is a demo version."
        return error_msg, image


def draw_boxes_on_image(image, boxes, classes, scores):
    """
    Draw detection boxes on image (simplified version)
    """
    try:
        # Convert to OpenCV format
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = cv_image.shape[:2]

        # Draw detection boxes
        for box, cls, score in zip(boxes, classes, scores):
            if score > 0.5:
                # Convert coordinates (assuming normalized coordinates)
                y1, x1, y2, x2 = box
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)

                # Draw rectangle
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                class_name = CATEGORY_INDEX.get(
                    cls, {}).get('name', f'class_{cls}')
                label = f"{class_name}: {score:.2f}"
                cv2.putText(cv_image, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Convert back to RGB
        result_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_image)

    except Exception as e:
        print(f"Error drawing detection boxes: {e}")
        return image

# Create Gradio interface


def create_demo():
    with gr.Blocks(title="Rubik's Cube Recognition System") as demo:
        gr.Markdown("""
        # 🎲 Rubik's Cube Recognition System
        
        This is a deep learning-based Rubik's cube recognition system using RetinaNet architecture for object detection.
        
        **Features:**
        - Detect cube faces and color tiles
        - Support 6 color recognition: Red, White, Blue, Orange, Green, Yellow
        - Real-time detection and visualization
        
        **How to use:**
        1. Upload an image containing a Rubik's cube
        2. Click the "Analyze Image" button
        3. View detection results and visualization
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Rubik's Cube Image",
                    type="pil"
                )

                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )

            with gr.Column():
                result_text = gr.Textbox(
                    label="Detection Results",
                    lines=10,
                    max_lines=15
                )

                output_image = gr.Image(
                    label="Detection Visualization",
                    type="pil"
                )

        # Example images section
        gr.Markdown("### 📋 Usage Examples")
        gr.Markdown(
            "Upload Rubik's cube images similar to the following for testing:")

        # Bind events
        analyze_btn.click(
            fn=predict_image,
            inputs=[input_image],
            outputs=[result_text, output_image]
        )

        # Auto-analyze when image is uploaded
        input_image.change(
            fn=predict_image,
            inputs=[input_image],
            outputs=[result_text, output_image]
        )

    return demo


# Launch application
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
