"""
RetinaNet configuration for Rubik's Cube detection.
"""

from official.vision.configs import retinanet
from official.vision.configs.common import backbone

HEIGHT = 640
WIDTH = 640
BATCH_SIZE = 8
NUM_CLASSES = 7  # 6 colors + face

def get_config():
    """Get the default configuration for RetinaNet training."""
    config = retinanet.retinanet_spinenet_mobile_coco()
    
    # Model config
    config.model.input_size = [HEIGHT, WIDTH, 3]
    config.model.num_classes = NUM_CLASSES + 1  # Add background class
    config.model.backbone.spinenet_mobile.model_id = '49'
    config.model.detection_generator.tflite_post_processing.max_classes_per_detection = NUM_CLASSES + 1
    
    # Training data config
    config.train_data.input_path = 'tfrecords/train-*.tfrecord'
    config.train_data.dtype = 'float32'
    config.train_data.global_batch_size = BATCH_SIZE
    config.train_data.parser.aug_scale_max = 1.0
    config.train_data.parser.aug_scale_min = 1.0
    
    # Validation data config
    config.validation_data.input_path = 'tfrecords/valid-*.tfrecord'
    config.validation_data.dtype = 'float32'
    config.validation_data.global_batch_size = BATCH_SIZE
    
    # Training config
    config.trainer.train_steps = 10000
    config.trainer.validation_steps = 100
    config.trainer.steps_per_loop = 100
    config.trainer.summary_interval = 100
    config.trainer.checkpoint_interval = 100
    config.trainer.validation_interval = 100
    
    # Optimizer config
    config.trainer.optimizer_config.learning_rate.type = 'cosine'
    config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
    config.trainer.optimizer_config.learning_rate.cosine.decay_steps = 10000
    config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05
    config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
    
    return config 