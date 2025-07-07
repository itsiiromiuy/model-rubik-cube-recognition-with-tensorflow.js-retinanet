---
title: Rubiks Cube Recognition
emoji: 🎲
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎲 Rubik's Cube Recognition with TensorFlow RetinaNet

This is a deep learning model that can recognize and analyze Rubik's cubes in images. The model is built using TensorFlow and RetinaNet architecture for object detection.

## 🚀 Features

- **Real-time Detection**: Upload images of Rubik's cubes for instant analysis
- **Multi-class Recognition**: Detect cube faces and 6 different color tiles
- **Interactive Interface**: Simple and intuitive Gradio web interface
- **Advanced AI**: Powered by RetinaNet with SpineNet-49 backbone

## 🎯 How to Use

1. **Upload**: Click "Upload Rubik's Cube Image" and select your image
2. **Analyze**: Click "🔍 Analyze Image" or wait for automatic processing
3. **Results**: View detection results and visualization with bounding boxes

## 🔬 Technical Details

- **Framework**: TensorFlow 2.15+ with Gradio interface
- **Architecture**: RetinaNet with SpineNet-49 backbone
- **Input Size**: 640×640 pixels
- **Classes**: 7 total (1 face + 6 color tiles)
- **Colors Detected**: Red, White, Blue, Orange, Green, Yellow

## 🌟 Model Architecture

### RetinaNet-SpineNet-49
- **Base Model**: RetinaNet for object detection
- **Backbone**: SpineNet-49 for feature extraction
- **Input Resolution**: 640×640×3
- **Output**: Bounding boxes with class predictions and confidence scores

### Detection Classes
1. `face` - Rubik's cube face
2. `red_tile` - Red color tile
3. `white_tile` - White color tile  
4. `blue_tile` - Blue color tile
5. `orange_tile` - Orange color tile
6. `green_tile` - Green color tile
7. `yellow_tile` - Yellow color tile

## 📊 Performance Metrics

| Metric | Target | Status |
|--------|--------|--------|
| mAP@0.5 | >0.85 | In Training |
| Inference Speed | <100ms | Optimized |
| Accuracy | >90% | Evaluating |

## 🛠️ Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/itsyuimorii/rubiks-cube-recognition
cd rubiks-cube-recognition

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📁 Project Structure

```
rubiks-cube-recognition/
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── configs/                 # Model configurations
│   └── retinanet_config.py
├── src/                     # Source code
│   ├── data/               # Data processing utilities
│   └── model/              # Model training and inference
└── images/                 # Training and test datasets
    ├── train/              # Training images and annotations
    ├── test/               # Test images and annotations
    └── valid/              # Validation images and annotations
```

## 🎮 Demo Status

⚠️ **Note**: This is a demo version. The complete trained model is currently being developed. The interface will show a preview of the detection capabilities.

## 📝 Dataset Information

- **Format**: COCO annotation format
- **Image Size**: 640×640 pixels
- **Training Images**: 50+ annotated cube images
- **Classes**: 7 object classes (face + 6 colors)
- **Annotation Tool**: LabelMe

## 🔧 Training Pipeline

```python
# Training command
python src/model/trainer.py --config configs/retinanet_config.py

# Inference command  
python src/model/visualize.py --image path/to/cube_image.jpg
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional training data
- Model optimization
- UI/UX enhancements
- Performance improvements

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **TensorFlow Model Garden** - RetinaNet implementation
- **SpineNet** - Backbone architecture
- **Gradio** - Web interface framework
- **Hugging Face** - Model hosting and deployment

## 📧 Contact

- **GitHub**: [@itsyuimorii](https://github.com/itsyuimorii)
- **Hugging Face**: [@itsyuimorii](https://huggingface.co/itsyuimorii)

## 🔗 References

- [RetinaNet Paper](https://arxiv.org/abs/1708.02002)
- [SpineNet Architecture](https://arxiv.org/abs/1912.05027)
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [LabelMe Annotation Tool](https://github.com/wkentaro/labelme)

---

*🎲 Ready to solve your Rubik's cube detection challenges!*



