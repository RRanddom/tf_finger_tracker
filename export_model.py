import os 
import numpy as np 
import tensorflow as tf
##from model.fused_model import main_network
# model/keypoints_heatmaps_model.py
from model.keypoints_heatmaps_model import main_network

image_height = 480 
image_width  = 640 

model_base_dir = "/data/FingerTipDataset/train/" #"/data/train_receipt/train_dir/keypoints_heatmap/"
# meta_file_name = "model.ckpt-10209.meta"
#model.ckpt-42120.meta
ckpt_file = "model.ckpt-42120"

def freeze_graph():
    input_placeholder = tf.placeholder(tf.float32, shape=[1, image_height, image_width, 3], name="input")
    # ...
    _ = main_network(input_placeholder)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(model_base_dir, ckpt_file))

    names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for name in names:
        print (name)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        ["heatmaps", "keypoints_pred"])

    output_graph = "frozen_model.pb"
    output_path = os.path.join(model_base_dir, output_graph)
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph" % len(output_graph_def.node))
    
    return None


if __name__ == "__main__":
    freeze_graph()

