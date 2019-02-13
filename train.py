import os
import numpy as np
import tensorflow as tf
from finger_dataset import get_dataset_split

from model.keypoints_heatmaps_model import keypoints_heatmaps_model,main_network
from functools import partial
from PIL import Image,ImageDraw
from glob import glob
import cv2

from utils.preprocessing import preprocess_image_and_points

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_logdir', '/data/FingerTipDataset/train',
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_string("pretrained_model", "pretrained_model/mobilenet_v2/mobilenet_v2_0.5_224.ckpt",
                    "Where the pretrained model stored")

flags.DEFINE_float('base_learning_rate', 3e-5,
                   'The base learning rate for model training.')

flags.DEFINE_integer('train_batch_size', 16,
                     'The number of images in each batch during training.')

flags.DEFINE_integer('log_steps', 20,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_summaries_steps', 100,
                     'How often, in seconds, we compute the summaries.')

def input_pipeline(split, num_epochs=1):
    is_training = (split == 'train')
    dataset = get_dataset_split(split)

    _preprocess_fn = partial(preprocess_image_and_points, is_training=is_training)
    dataset = dataset.map(_preprocess_fn)

    batch_size = (FLAGS.train_batch_size if is_training else 1)
    dataset = dataset.shuffle(buffer_size=500).repeat(num_epochs).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def vis_input_data():
    image, bbox, points = input_pipeline("train", num_epochs=1)

    print ("what is image:{}".format(image.shape))
    sess = tf.Session()

    while True:
        img_nd, bbox_nd, points_nd = sess.run([image, bbox, points])

        img_nd = img_nd[0].astype('uint8')
        bbox_nd = bbox_nd[0]
        points_nd = points_nd[0]

        bbox_nd[:,0] *= 640
        bbox_nd[:,1] *= 480

        points_nd[:,0] *= 640
        points_nd[:,1] *= 480

        bbox_nd = np.round(bbox_nd).astype('int')
        points_nd = np.round(points_nd).astype('int')

        img_nd = cv2.cvtColor(img_nd, cv2.COLOR_RGB2BGR)

        cv2.rectangle(img_nd, tuple(bbox_nd[0]), tuple(bbox_nd[1]), color=(0,0,255))
        cv2.circle(img_nd, tuple(points_nd[0]) ,3, (0,255,33), -1)
        cv2.circle(img_nd, tuple(points_nd[1]) ,3, (0,255,33), -1)
        cv2.circle(img_nd, tuple(points_nd[2]) ,3, (0,255,33), -1)

        print ("heyhey:{}".format(img_nd.shape))

        cv2.imshow('frame', img_nd)
        cv2.waitKey(100)

def vis_infer_result():
    train_dir = FLAGS.train_logdir
    img = tf.placeholder(tf.float32, shape=[1, 480, 640])
    keypoints_prediction, heatmaps_prediction, keypoints, heatmaps = main_network(img)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(train_dir))

    for img_name in glob("/data/FingerTipDataset/I_Avenue/*.png"):
        # im_pil = Image.open(img_name)
        mat = cv2.imread(img_name)
        mat_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        mat_rgb = np.expand_dims(mat_rgb, 0)
        kpd = sess.run(keypoints, feed_dict={img:mat_rgb})
        print (kpd)



if __name__ == "__main__":

    # vis_input_data()
    vis_infer_result()

    # tf.logging.set_verbosity(tf.logging.INFO)

    # run_config = tf.estimator.RunConfig()\
    #                 .replace(save_summary_steps=FLAGS.save_summaries_steps)\
    #                 .replace(log_step_count_steps=FLAGS.log_steps)
    # decay_factor = .9

    # train_dir = FLAGS.train_logdir

    # width = 640
    # height = 480

    # params = {
    #     "width" : 640,
    #     "height" : 480,
    #     "depth_multiplier" : .5,
    #     "train_dir" : train_dir,
    #     "learning_rate" : FLAGS.base_learning_rate,
    #     "pretrained_model" : FLAGS.pretrained_model
    # }

    # num_of_training_epochs = 5
    # model = tf.estimator.Estimator(
    #     model_fn = keypoints_heatmaps_model,
    #     model_dir = train_dir,
    #     config = run_config,
    #     params = params
    # )

    # for epoch in range(num_of_training_epochs):
    #     tf.logging.info("Starting a training cycle.")
    #     model.train(input_fn=lambda : input_pipeline('train'))
    #     lr = params["learning_rate"] * decay_factor
    #     params.update({"learning_rate" : lr})
    #     tf.logging.info("Starting to evaluate.")
    #     model.evaluate(input_fn=lambda : input_pipeline('eval'))
