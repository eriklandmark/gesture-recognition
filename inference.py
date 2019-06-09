import cv2
import generators
import numpy as np
import tensorflow as tf
import os

with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], "exported_model")
    labels = generators.get_labels()
    # camera = cv2.VideoCapture(0)
    # frame_buffer = []
    # for i in range(45):
    #     return_value, frame = camera.read()
    #     frame_buffer.append(cv2.resize(frame, (176, 100), interpolation=cv2.INTER_LINEAR))
    frame_buffer = generators.parse_video_frames_from_path(os.path.join(os.getcwd(), "recording"))
    examples = np.asarray(frame_buffer)
    examples = examples.astype('float32')
    np.divide(examples, 255.0, examples)
    scores = sess.run(sess.graph.get_tensor_by_name('dense_1/Softmax:0'), feed_dict={
        "input_1:0": np.expand_dims(examples, axis=0)
    })
    score = np.argmax(scores[0])

    print(labels[score], score)

