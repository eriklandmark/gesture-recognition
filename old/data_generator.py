import os
import tensorflow as tf
import cv2
import csv
import time
import threading

DATASET_PATH = "D:/Datasets/20bn-jester/extracted"
LABELS_PATH = "D:/Datasets/20bn-jester/jester-v1-labels.csv"
TRAIN_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-train.csv"
tf.enable_eager_execution()


# {'176': 108197, '132': 38523, '100': 857, '106': 43, '142': 353, '122': 49, '160': 50, '172': 20}


def get_labels():
    # csv_reader = csv.reader(LABELS_PATH, delimiter=';')
    with open(LABELS_PATH, "r") as file:
        return list(map(lambda x: x.strip(), file.readlines()))


class BatchThread(threading.Thread):
    def __init__(self, threadID, train_ids, labels, batch_id, batch_size):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.train_ids = train_ids
        self.labels = labels
        self.batch_id = batch_id
        self.batch_size = batch_size

    def run(self):
        batch = self.train_ids[self.batch_id * self.batch_size: (self.batch_id * self.batch_size) + self.batch_size]
        path = "D:/Datasets/20bn-jester/records_16/train.tfrecord" + '.{:04d}'.format(self.batch_id)
        with tf.python_io.TFRecordWriter(path) as writer:
            for (video_id, label) in batch:
                writer.write(tf.train.Example(features=tf.train.Features(feature={
                    'image_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(video_id)])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.labels.index(label)])),
                    'image_raw': create_vid_feature(video_id)
                })).SerializeToString())


def gen_tffiles():
    with open(TRAIN_LIST_PATH, "r") as file:
        labels = get_labels()
        csv_reader = csv.reader(file, delimiter=';')
        batch_size = 16
        train_ids = [line for line in csv_reader]
        num_batches = int(len(train_ids) / batch_size)
        batch_id = 0
        starting_time = time.time()
        threads = 16
        active_threads = []

        while batch_id < num_batches - threads:
            for thread_id in range(threads):
                thread = BatchThread(thread_id, train_ids, labels, batch_id, batch_size)
                batch_id += 1
                thread.start()
                active_threads.append(thread)

            for thread in active_threads:
                thread.join()

            active_threads = []
            # current_time = time.time()
            # time_per_batch = (current_time - starting_time) / (batch_id - 1)
            # time_left = (num_batches - batch_id - 1) * time_per_batch
            # time_parsed = f"Time left: {int(time_left / 3600)} hours and {int((time_left / 3600 - int(time_left / 3600)) * 60)} min ({time_per_batch} s/b)"

            print(f"Processed batch {batch_id - 1} of {num_batches} ({'{0:.{1}f}'.format((batch_id - 1 / num_batches) * 100, 4)} %)")

        current_time = time.time()
        total_time = current_time - starting_time
        print(f"Generated {num_batches} in {int(total_time / 3600)} hours and {(total_time / 3600 - int(total_time / 3600)) * 60}")


def create_vid_feature(vid_id):
    video_path = os.path.join(DATASET_PATH, vid_id)
    vid_frames_path = os.listdir(video_path)
    vid_frames = []
    for frame_nr in vid_frames_path:
        image = cv2.imread(os.path.join(video_path, frame_nr))
        _, width, _ = image.shape
        if not width == 176:
            image = cv2.resize(image, (176, 100), interpolation=cv2.INTER_LINEAR)
        vid_frames.append(image)

    padding_frames = 45 - len(vid_frames)
    for i in range(int(padding_frames / 2)):
        vid_frames.insert(0, vid_frames[0])
    for i in range(padding_frames - int(padding_frames / 2)):
        vid_frames.append(vid_frames[-1])

    bytes_list = []
    for frame in vid_frames:
        bytes_list.append(frame.tostring())

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))


gen_tffiles()

def old_generator():
    while True:
        for batch in os.listdir("D:/Datasets/20bn-jester/records"):
            tf.logging.info("Processing batch: " + batch)
            record_iterator = tf.python_io.tf_record_iterator(
                path=os.path.join("D:/Datasets/20bn-jester/records", batch))
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
                    restored_frames.append(reconstructed_img.tolist())

                examples.append(restored_frames)
                labels.append(label)

            examples_tensor = tf.convert_to_tensor(examples)
            examples_tensor = tf.cast(examples_tensor, tf.float32)
            examples_tensor = tf.math.divide(examples_tensor, 255.0)

            categorical_labels = [tf.keras.utils.to_categorical(label, num_classes=27, dtype='float32').tolist() for
                                  label in labels]
            yield (examples_tensor, [categorical_labels])