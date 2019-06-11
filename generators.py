import tensorflow as tf
import numpy as np
import os
import cv2
import csv
import lib

DATASET_PATH = "D:/Datasets/20bn-jester/extracted"
# LABELS_PATH = "D:/Datasets/20bn-jester/jester-v1-labels.csv"
LABELS_PATH = os.path.join(os.getcwd(), "test_material/labels.txt")
TRAIN_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-train.csv"


class DataGeneratorV1(tf.keras.utils.Sequence):
    def __init__(self, data_ids, labels, batch_size=16, shuffle=False, verbose=False):
        self.batch_size = batch_size
        self.labels = labels
        self.data_ids = data_ids
        self.shuffle = shuffle
        self.verbose = verbose

    def __len__(self):
        return int(np.floor(len(self.data_ids) / self.batch_size))

    def __getitem__(self, batch_id):
        batch = self.data_ids[batch_id * self.batch_size: (batch_id * self.batch_size) + self.batch_size]
        if self.verbose:
            print(f"Creating batch {batch_id}")
        return self.create_batch(batch, self.labels)

    def create_batch(self, data_array, labels, verbose=False, prefix_path=DATASET_PATH):
        examples = []
        parsed_labels = []
        if verbose:
            lib.progress_bar(0, len(data_array), prefix="Created examples", suffix=f'Complete (0/{len(data_array)})',
                             length=30)
        for (index, (video_id, label)) in enumerate(data_array):
            vid_frames = [os.path.join(prefix_path, video_id, p) for p in os.listdir(os.path.join(prefix_path, video_id))]
            examples.append(_parse_video_frames(vid_frames, 45))
            parsed_labels.append(labels.index(label))

            if verbose:
                lib.progress_bar(index + 1, len(data_array), prefix="Creating examples",
                                 suffix=f'Completed ({index + 1}/{len(data_array)})', length=30)

        examples = np.asarray(examples)
        examples = examples.astype('float32')
        np.divide(examples, 255.0, examples)

        categorical_labels = np.asarray(
            [tf.keras.utils.to_categorical(label, num_classes=len(labels), dtype='float32') for
             label in parsed_labels])

        return examples, categorical_labels


class DataGeneratorV2(tf.keras.utils.Sequence):
    def __init__(self, data_ids, labels, batch_size=16, usable_frames=8, shuffle=False, verbose=False):
        self.batch_size = batch_size
        self.labels = labels
        self.data_ids = data_ids
        self.shuffle = shuffle
        self.verbose = verbose
        self.usable_frames = usable_frames

    def __len__(self):
        return int(np.floor(len(self.data_ids) / self.batch_size))

    def __getitem__(self, batch_id):
        batch = self.data_ids[batch_id * self.batch_size: (batch_id * self.batch_size) + self.batch_size]
        if self.verbose:
            print(f"Creating batch {batch_id}")
        return self.create_batch(batch, self.labels)

    def create_batch(self, data_array, labels, verbose=False, prefix_path=DATASET_PATH):
        examples = []
        parsed_labels = []
        if verbose:
            lib.progress_bar(0, len(data_array), prefix="Created examples", suffix=f'Complete (0/{len(data_array)})',
                             length=30)
        for (index, (video_id, label)) in enumerate(data_array):
            total_frames = os.listdir(os.path.join(prefix_path, video_id))
            idx = np.round(np.linspace(0, len(total_frames) - 1, self.usable_frames)).astype(int)
            vid_frames = [os.path.join(prefix_path, video_id, total_frames[i]) for i in idx]
            examples.append(_parse_video_frames(vid_frames, self.usable_frames))
            parsed_labels.append(labels.index(label))

            if verbose:
                lib.progress_bar(index + 1, len(data_array), prefix="Creating examples",
                                 suffix=f'Completed ({index + 1}/{len(data_array)})', length=30)

        examples = np.asarray(examples)
        examples = examples.astype('float32')
        np.divide(examples, 255.0, examples)

        categorical_labels = np.asarray(
            [tf.keras.utils.to_categorical(label, num_classes=len(labels), dtype='float32') for
             label in parsed_labels])

        return examples, categorical_labels


def _parse_video_frames(orig_vid_frames, seq_len):
    vid_frames = []
    for frame_nr in orig_vid_frames:
        image = cv2.imread(frame_nr)
        _, width, _ = image.shape
        if not width == 176:
            image = cv2.resize(image, (176, 100), interpolation=cv2.INTER_LINEAR)
        vid_frames.append(image)
    padding_frames = seq_len - len(vid_frames)
    if padding_frames > 0:
        for i in range(int(padding_frames / 2)):
            vid_frames.insert(0, vid_frames[0])
        for i in range(padding_frames - int(padding_frames / 2)):
            vid_frames.append(vid_frames[-1])
    return vid_frames


def get_labels():
    with open(LABELS_PATH, "r") as file:
        return list(map(lambda x: x.strip(), file.readlines()))


def get_data_ids_from_file(file):
    with open(file, "r") as file:
        csv_reader = csv.reader(file, delimiter=';')
        return [line for line in csv_reader]


if __name__ == "__main__":
    _labels = get_labels()
    _data_ids = get_data_ids_from_file(TRAIN_LIST_PATH)
    _generator_v2 = DataGeneratorV2(_data_ids, _labels)
    _data = _data_ids[1000:1200]
    print(_data)
    X, Y = _generator_v2.create_batch(_data, _labels, verbose=False)

