"""
Shared configurations and utilities for the Rubik's Cube detection project.
"""

import matplotlib.pyplot as plt
import numpy as np
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils
from PIL import Image

# Model configurations
HEIGHT, WIDTH = 640, 640
EXPORT_DIR = './exported_model/'

# Category definitions
category_index = {
    1: {'id': 1, 'name': 'face'},
    2: {'id': 2, 'name': 'red_tile'},
    3: {'id': 3, 'name': 'white_tile'},
    4: {'id': 4, 'name': 'blue_tile'},
    5: {'id': 5, 'name': 'orange_tile'},
    6: {'id': 6, 'name': 'green_tile'},
    7: {'id': 7, 'name': 'yellow_tile'}
}

# TensorFlow Example decoder
tf_ex_decoder = TfExampleDecoder()

def process_image(image_path):
    """
    Process an image for model input.
    
    Args:
        image_path: Path to the image file or PIL Image object
        
    Returns:
        Processed image tensor
    """
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
        
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        
    return image

def visualize_detection(image, boxes, classes, scores, category_index,
                       min_score_thresh=0.30, max_boxes_to_draw=20):
    """
    Visualize detection results.
    
    Args:
        image: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: float32 numpy array of shape [N, 4]
        classes: integer numpy array of shape [N]
        scores: float numpy array of shape [N]
        category_index: dict containing category information
        min_score_thresh: minimum score threshold for visualization
        max_boxes_to_draw: maximum number of boxes to visualize
        
    Returns:
        uint8 numpy array with shape (img_height, img_width, 3) with boxes drawn on it
    """
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)
    
    return image

def show_batch(raw_records, save_dir='examples'):
    """
    Show and save a batch of images with their annotations.
    
    Args:
        raw_records: TFRecord dataset
        save_dir: Directory to save the visualizations
    """
    plt.figure(figsize=(20, 20))
    for i, serialized_example in enumerate(raw_records):
        plt.subplot(1, 3, i + 1)
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = decoded_tensors['image'].numpy().astype('uint8')
        scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
        
        image = visualize_detection(
            image,
            decoded_tensors['groundtruth_boxes'].numpy(),
            decoded_tensors['groundtruth_classes'].numpy().astype('int'),
            scores,
            category_index)
            
        im = Image.fromarray(image)
        im.save(f'{save_dir}/batch_image_{i+1}.png')
