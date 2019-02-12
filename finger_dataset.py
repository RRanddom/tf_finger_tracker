"""
Convert Image/Edgemap data to TFRecord file format with Example protos
"""

import math
import os.path
import re
import sys
import tensorflow as tf
import random
import build_data
from glob import glob

MAX_NUM_SHARDS = 4

dataset_root = "/data/FingerTipDataset"
tfrecord_output_dir = "/data/FingerTipDataset/tfrecord"
annot_dir = "txt"

eval_parts = ["I_TennisField"]

def _get_files(dataset_split):
    annot_path = os.path.join(dataset_root, annot_dir)
    all_txts = glob(annot_path + "/*.txt")
    
    eval_annots = []
    train_annots = []

    for eval_name in eval_parts:
        for _txt_file in all_txts:
            if eval_name in _txt_file:
                eval_annots.append(_txt_file)

    train_annots = (set(all_txts) - set(eval_annots))

    annots = []
    if dataset_split == 'train':
        annots = train_annots
    else:
        annots = eval_annots

    valid_annots = []
    for annot_txt_file_name in annots:
        filenamebase = os.path.basename(annot_txt_file_name).split('.')[0]
        filenamebase = filenamebase[0:-6]
        if os.path.exists(os.path.join(dataset_root, filenamebase)):
            valid_annots.append(annot_txt_file_name)
    
    print ("valid annots:{}".format(valid_annots))
    return valid_annots
    

def _convert_dataset(dataset_split):
    """
    convert images and annots to tfrecord format.
    Args:
        dataset_split: "train" or "eval"
    """
    annot_files = _get_files(dataset_split)
    dirs = []
    num_images = 0
    for annot_txt_file_name in annot_files:
        filenamebase = os.path.basename(annot_txt_file_name).split('.')[0]
        filenamebase = filenamebase[0:-6]
        the_dir = os.path.join(dataset_root, filenamebase)
        dirs.append(the_dir)
        num_images += len(glob(the_dir+"/*.png"))

    if not os.path.exists(tfrecord_output_dir):
        os.mkdir(tfrecord_output_dir)
    
    image_reader = build_data.ImageReader("png", channels=3)

    _NUM_SHARDS = len(annot_files)
    annot_file_per_shard = 1
    if len(annot_files) > MAX_NUM_SHARDS:
        annot_file_per_shard = round(1.0 * len(annot_files) / MAX_NUM_SHARDS)
        _NUM_SHARDS = MAX_NUM_SHARDS

    start_index = 0
    image_index = 1

    for shard_id in range(_NUM_SHARDS):
        shard_filename = "%s-%02d-of-%02d.tfrecord" % (
            dataset_split, shard_id, _NUM_SHARDS)
        output_filename = os.path.join(tfrecord_output_dir, shard_filename)

        if shard_id == _NUM_SHARDS-1:
            end_index = len(annot_files) 
        else:
            end_index = start_index + annot_file_per_shard

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for _idx in range(start_index, end_index):
                annot_txt_file_name = annot_files[_idx]
                img_folder = dirs[_idx]
                
                annot_contents = open(annot_txt_file_name, 'r').readlines()
                for annot_content in annot_contents:
                    content_arr = annot_content.split()
                    img_name = content_arr[0]
                    pts = content_arr[1:-4]
                    pts = [float(_) for _ in pts]
                    bbox = pts[:4]
                    points = pts[4:]

                    img_full_path = os.path.join(img_folder, img_name)
                    image_data = tf.gfile.GFile(img_full_path, 'rb').read()
                    height, width = image_reader.read_image_dims(image_data)
                    
                    example = build_data.image_label_to_tfexample(
                        image_data, img_name, height, width, bbox, points)

                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (image_index, num_images, shard_id))
                    sys.stdout.flush()

                    image_index += 1
                    tfrecord_writer.write(example.SerializeToString())

        start_index = end_index
    
        sys.stdout.write('\n')
        #sys.stdout.flush()

def main(unused_argv):
  # Only support converting 'train' and 'eval' sets for now.
  for dataset_split in ['train', 'eval']:
      _convert_dataset(dataset_split)


def _parse_function(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        "image/bbox": tf.FixedLenFeature(
            (2,), tf.float32, default_value=None),
        'image/points': tf.FixedLenFeature(
            (6,), tf.float32, default_value=None),
    }

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    with tf.variable_scope('decoder'):
        input_image = tf.image.decode_image(parsed_features["image/encoded"], channels=3),
        file_name = parsed_features["image/filename"]
        input_height = parsed_features["image/height"]
        input_width = parsed_features["image/width"]
        bbox = parsed_features["image/bbox"]
        points = parsed_features["image/points"]

    return input_image, file_name, input_height, input_width, bbox, points


def get_dataset_split(split_name):
    dataset_dir = tfrecord_output_dir
    file_pattern = 'train*.tfrecord' if split_name=='train' else 'eval*.tfrecord'
    filenames = tf.gfile.Glob(os.path.join(dataset_dir, file_pattern))
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset.map(_parse_function)

if __name__ == '__main__':
    tf.app.run()
