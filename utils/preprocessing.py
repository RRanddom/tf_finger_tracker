import tensorflow as tf 
from utils.preprocess_utils import random_left_right_flip ,random_crop

#def _preprocess_fn(input_image, image_name, input_height, input_width, points, is_training=False):

def preprocess_image_and_points(input_image, image_name, height, width, bbox, points, is_training=True):
    """Preprocesses the image and points.

    Args:
        input_image : image tensor. shape of [height, width, 3]
        image_name  : string tensor. name of the image.
        height: image height
        width : image width
        points: 4 points. ~
        is_training : If the preprocessing is used for training or not.

    Returns:
        image_processed,
        bbox_processed,
        points_processed,
        
    Raises:
    """
    # input_image.set_shape([height, width, 3])
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    input_image = tf.cast(input_image, tf.float32)
    points = tf.reshape(points, [3,2])
    points = tf.cast(points, tf.float32)

    bbox = tf.reshape(bbox, [2,2])
    bbox = tf.cast(bbox, tf.float32)

    input_image, bbox, points = random_crop(input_image, bbox, points)
    
    if is_training:
        input_image, bbox, points = random_left_right_flip(input_image, bbox, points, width, height)

    return input_image, bbox, points
