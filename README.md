---
title: Rubik's Cube Recognition with TensorFlow.js RetinaNet
emoji: 🎲
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎲 Rubik's Cube Recognition with TensorFlow.js RetinaNet

This project provides a deep learning solution for recognizing Rubik's Cube patterns using TensorFlow.js and RetinaNet-SpineNet-49.

## 🌟 Model Description

The model is based on RetinaNet with SpineNet-49 backbone, trained to detect:
- Rubik's cube faces
- Individual color tiles (red, white, blue, orange, green, yellow)

### Model Architecture
- Base Model: RetinaNet-SpineNet-49
- Input Size: 640x640x3
- Number of Classes: 7 (6 colors + face)
- Output: Bounding boxes with class predictions

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| mAP    | TBD   |
| FPS    | TBD   |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://huggingface.co/spaces/[your-username]/rubiks-cube-recognition

# Install dependencies
pip install -r requirements.txt

# Run the demo
python app.py
```

## 📦 Project Structure

```
rubiks-cube-recognition/
├── app.py                 # Gradio web interface
├── src/
│   ├── data/             # Data processing utilities
│   │   ├── labelme2coco.py
│   │   └── shared.py
│   ├── model/            # Model training and inference
│   │   ├── trainer.py
│   │   └── visualize.py
│   └── utils/            # Utility functions
├── configs/              # Model configurations
├── examples/             # Example images and results
└── requirements.txt      # Project dependencies
```

## 🔧 Usage

### Training

```python
python src/model/trainer.py --config configs/retinanet_config.py
```

### Inference

```python
python src/model/visualize.py --image path/to/image.jpg
```

## 📝 Dataset

The model is trained on a custom dataset of Rubik's cube images, annotated with:
- Face detection
- Color tile detection

### Data Format
- Annotations: COCO format
- Image size: 640x640
- Classes: 7 (face, red_tile, white_tile, blue_tile, orange_tile, green_tile, yellow_tile)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- TensorFlow Model Garden
- RetinaNet Implementation
- SpineNet Architecture

## 📧 Contact

- GitHub: https://github.com/itsiiromiuy
- Hugging Face:  https://huggingface.co/itsyuimorii

## 📚 Citation

```bibtex
@software{rubiks_cube_recognition,
  title = {Rubik's Cube Recognition with TensorFlow.js RetinaNet},
  author = {[Hedgehog, WildStriker]},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/spaces/[your-username]/rubiks-cube-recognition}
}
```
 
