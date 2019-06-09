import tensorflow as tf
import numpy as np
import cv2

record_iterator = tf.python_io.tf_record_iterator(path="D:/Datasets/20bn-jester/records_16/train.tfrecord.1232")
examples = []
labels = []

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    label = int(example.features.feature['label'].int64_list.value[0])
    frames = example.features.feature['image_raw'].bytes_list.value

    restored_frames = []
    for frame in frames:
        img_1d = np.fromstring(frame, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((100, 176, 3))
        restored_frames.append(reconstructed_img)

    examples.append(restored_frames)
    labels.append(label)

    cv2.imshow('image', restored_frames[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(len(examples))
#examples_tensor = tf.Variable(examples)
#examples_tensor = tf.cast(examples_tensor, tf.float32)
#examples_tensor = tf.math.divide(examples_tensor, 255.0)

#print(examples_tensor.shape)

categorical_labels = [tf.keras.utils.to_categorical(label, num_classes=27, dtype='float32').tolist() for label in labels]
label_tensor = tf.Variable(categorical_labels)
print(label_tensor.shape)