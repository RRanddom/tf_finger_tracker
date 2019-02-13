tflite_convert \
    --output_file=finger_track.tflite \
    --graph_def_file=frozen_model.pb \
    --input_arrays=input \
    --output_arrays=heats_map_regression/pred_keypoints/BiasAdd


