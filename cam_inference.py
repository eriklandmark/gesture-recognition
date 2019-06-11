import cv2
import generators
import numpy as np
import tensorflow as tf

GPU_DEVICE = 0

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.device('/device:GPU:' + str(GPU_DEVICE)):
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], "exported_models/v3")

    labels = generators.get_labels()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    frame_buffer = []

    usable_frames = 16

    while True:
        return_value, frame = camera.read()
        if return_value:
            if len(frame_buffer) <= usable_frames:
                p_frame = cv2.resize(frame, (176, 100), interpolation=cv2.INTER_LINEAR)
                # p_frame = np.asarray(p_frame)
                p_frame = p_frame.astype('float32')
                np.divide(p_frame, 255.0, p_frame)
                frame_buffer.append(p_frame)
            if len(frame_buffer) > usable_frames:
                frame_buffer.pop(0)
            if len(frame_buffer) == usable_frames:
                with tf.device('/device:GPU:' + str(GPU_DEVICE)):
                    scores = sess.run(sess.graph.get_tensor_by_name('dense/Softmax:0'), feed_dict={
                        "conv3d_input:0": np.expand_dims(frame_buffer, axis=0)
                    })
                gesture = np.argmax(scores)
                score = scores[0][gesture]
                if score > 0.5 and gesture != 25 and gesture != 26:
                    print(labels[gesture], score, gesture)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cv2.destroyAllWindows()

