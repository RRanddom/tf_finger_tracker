import os
import cv2
import numpy as np
import tensorflow as tf

pb_file_path = 'frozen_model.pb'

def main():

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # init the graph
    sess = tf.Session()

    with tf.gfile.GFile(pb_file_path, 'rb') as f:
      graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    input_image = sess.graph.get_tensor_by_name("input:0")
    heatmaps = sess.graph.get_tensor_by_name("heatmaps:0")
    keypoints = sess.graph.get_tensor_by_name("keypoints_pred:0")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[:,160:960+160,:]
        frame = cv2.resize(frame, (640, 480))
        img_for_network = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        keypoints_nd, heatmaps_nd = sess.run([keypoints, heatmaps], feed_dict={input_image: np.expand_dims(img_for_network, 0)})
        keypoints_nd = keypoints_nd[0]
        keypoints_nd *= np.array([640.0, 480.0])
        keypoints_nd = keypoints_nd.astype('int32')
        points = [tuple(point) for point in  list(keypoints_nd)]
        
        p1, p2, p3 = points

        cv2.circle(frame, p1, 3, (0,255,0), thickness=-1)
        cv2.circle(frame, p2, 3, (0,255,0), thickness=-1)
        cv2.circle(frame, p3, 3, (0,255,0), thickness=-1)

        heatmap = cv2.resize(heatmaps_nd[0], (640, 480))
        cv2.imshow('heatmap', heatmap)
        cv2.imshow("kacha", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # cv2.line(frame, p1, p2, [0,255,0], 3)
        # cv2.line(frame, p2, p4, [0,255,0], 3)
        # cv2.line(frame, p4, p3, [0,255,0], 3)

if __name__ == "__main__":
    main()