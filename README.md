### 🤖 Project Summary: AI Rubik's Cube Recognition Using TensorFlow.js and RetinaNet-SpineNet-49 

This project utilizes **TensorFlow.js** and the **RetinaNet-SpineNet-49** model (`retinanet_spinenet_mobile_coco`) to train an AI system that recognizes and interprets Rubik's Cube patterns via a camera. By integrating **computer vision** 🖼️, **COCO annotations** 🗂️, and **state-of-the-art object detection models** 🚀, the system detects and identifies each face and tile color of a Rubik's Cube, paving the way for automated solving.

### 📝 Description 

The project is structured into three core components:

1. **Data Preparation** 🖌️:
   - Collected Rubik's Cube images annotated using **LabelMe**.
   - Converted annotations from **LabelMe JSON** format to **COCO JSON** format using a custom Python script.
   - Defined detection categories, including color tiles (`red_tile`, `white_tile`, `blue_tile`, etc.).

2. **Model Training** 🏋️‍♀️:
   - Trained the **RetinaNet-SpineNet-49** model (`retinanet_spinenet_mobile_coco`) using COCO-formatted annotations.
   - Fine-tuned the model for real-time classification and localization of Rubik's Cube tiles.

3. **Visualization and Evaluation** 👀:
   - Developed a visualization script to overlay predictions on test images with bounding boxes and labels.
   - Exported results to PNG format for validation and analysis.

### ✨ Key Features 
- **Advanced Object Detection** 🎯: High-precision Rubik's Cube tile detection using the RetinaNet-SpineNet-49 model.
- **Dynamic Tile Recognition** 🕹️: Real-time identification of cube tiles and colors via camera input.
- **Streamlined Annotation Workflow** 🔄: Seamless conversion of LabelMe JSON annotations into the COCO JSON format.
- **Custom Visualization Tools** 🖼️: Debugging and enhancing predictions through overlayed visual outputs.
 
---

## ⚙️ How It Works 

1. **Data Collection**: Annotate Rubik's Cube images using LabelMe. ✍️  
2. **COCO Conversion**: Convert annotations with the `labelme_to_coco.py` script. 🔄  
3. **Model Training**:  
   - Configure the `retinanet_spinenet_mobile_coco` model using TensorFlow Model Garden.  
   - Train the model on a custom Rubik's Cube dataset. 🧠  
4. **Deployment**: Use TensorFlow.js to deploy the trained model for real-time detection. 🌐  

---

## 🚀 Getting Started 

### Prerequisites 🛠️
- Python 3.8+
- TensorFlow
- TensorFlow.js
- TensorFlow Model Garden
- LabelMe for annotation

### Installation 📦

1. Clone the repository:
   ```bash
   git clone [repository_link]
   cd rubiks-cube-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Annotate your Rubik's Cube images using LabelMe and place them in the `images/` folder.

---

### Usage 💻

1. **Convert Annotations**:
   ```bash
   python labelme_to_coco.py
   ```

2. **Train the Model**:
   - Modify the configuration for `retinanet_spinenet_mobile_coco`.
   - Train the model:
     ```bash
     python train.py --model=retinanet_spinenet_mobile_coco --config=configs/retinanet_spinenet_mobile_coco.config --data_dir=path_to_coco_data
     ```

3. **Run Detection**:
   - Convert the trained model to TensorFlow.js format.
   - Deploy the model for real-time detection.

---

## 📊 Results 

Sample visualizations with bounding boxes and labels are saved in the `outputs/` folder. 🖼️

---

## 🛤️ Roadmap 
- 🤖 Integrate a Rubik's Cube-solving algorithm.
- 📈 Expand the dataset to improve detection accuracy.
- 🌐 Deploy the system as a web-based application.

---

## 🤝 Contributions 
Contributions are welcome! Feel free to submit issues or pull requests for improvements. 💡

---

This version adds emojis to make the README more engaging and visually appealing! Let me know if you need further changes!
