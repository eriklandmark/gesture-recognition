import cv2
import generators
import numpy as np
import tensorflow as tf


with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], "exported_model")

    labels = generators.get_labels()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    frame_buffer = []

    while True:
        return_value, frame = camera.read()
        if return_value:
            if len(frame_buffer) <= 45:
                frame = cv2.resize(frame, (176, 100), interpolation=cv2.INTER_LINEAR)
                frame = np.asarray(frame)
                frame = frame.astype('float32')
                np.divide(frame, 255.0, frame)
                frame_buffer.append(frame)
            if len(frame_buffer) > 45:
                frame_buffer.pop(0)
            if len(frame_buffer) == 45:
                scores = sess.run(sess.graph.get_tensor_by_name('dense_1/Softmax:0'), feed_dict={
                    "input_1:0": np.expand_dims(frame_buffer, axis=0)
                })
                score = np.argmax(scores)
                print(labels[score])
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cv2.destroyAllWindows()

