import gradio as gr
import numpy as np
from PIL import Image
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

def predict_image(image):
    """
    Make predictions on input image - Demo version
    """
    if image is None:
        return "Please upload an image", None

    try:
        # Convert to PIL image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Get image information
        width, height = image.size
        
        # Demo response since model is not trained yet
        result_text = f"""🎲 Rubik's Cube Analysis Results

📊 Image Information:
- Dimensions: {width} × {height} pixels
- Format: {getattr(image, 'format', 'PIL Image')}

🔍 Detection Status:
✅ Image uploaded successfully
✅ Image format is valid
⚠️  AI model is currently in development

📝 Demo Mode:
This is a preview of the Rubik's cube recognition system.
The complete RetinaNet model will detect:

🎯 Target Detection Classes:
- Cube faces
- Red tiles
- White tiles  
- Blue tiles
- Orange tiles
- Green tiles
- Yellow tiles

🚀 Coming Soon:
- Real-time object detection
- Bounding box visualization
- Confidence scores
- 3D cube state analysis
"""

        return result_text, image

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}\n\nThis is a demo version."
        return error_msg, image

def create_demo():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="🎲 Rubik's Cube Recognition System",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎲 Rubik's Cube Recognition System</h1>
            <p style="font-size: 18px; color: #666;">
                Deep Learning-based Rubik's Cube Detection using RetinaNet Architecture
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                
                input_image = gr.Image(
                    label="Upload Rubik's Cube Image",
                    type="pil",
                    height=350
                )

                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💡 Tips
                - Upload clear images of Rubik's cubes
                - Good lighting recommended
                - JPG/PNG formats supported
                """)

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                
                result_text = gr.Textbox(
                    label="Detection Report",
                    lines=12,
                    max_lines=15,
                    show_copy_button=True
                )

                output_image = gr.Image(
                    label="Processed Image",
                    type="pil",
                    height=350
                )

        # Event handlers
        analyze_btn.click(
            fn=predict_image,
            inputs=[input_image],
            outputs=[result_text, output_image]
        )

        input_image.change(
            fn=predict_image,
            inputs=[input_image],
            outputs=[result_text, output_image]
        )

        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #eee;">
            <p><strong>🔬 Technology Stack:</strong> TensorFlow • RetinaNet • SpineNet-49 • Gradio</p>
            <p><strong>📧 Contact:</strong> <a href="https://huggingface.co/itsyuimorii">@itsyuimorii</a></p>
        </div>
        """)

    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
