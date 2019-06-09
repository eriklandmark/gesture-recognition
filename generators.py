import tensorflow as tf
import numpy as np
import os
import cv2
import csv
import lib

DATASET_PATH = "D:/Datasets/20bn-jester/extracted"
# LABELS_PATH = "D:/Datasets/20bn-jester/jester-v1-labels.csv"
LABELS_PATH = os.path.join(os.getcwd(), "labels.txt")
TRAIN_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-train.csv"


class DataGenerator(tf.keras.utils.Sequence):
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
        return create_batch(batch, self.labels)


def parse_video_frames_from_path(path):
    vid_frames_path = os.listdir(path)
    vid_frames = []
    for frame_nr in vid_frames_path:
        image = cv2.imread(os.path.join(path, frame_nr))
        _, width, _ = image.shape
        if not width == 176:
            image = cv2.resize(image, (176, 100), interpolation=cv2.INTER_LINEAR)
        vid_frames.append(image)
    padding_frames = 45 - len(vid_frames)
    for i in range(int(padding_frames / 2)):
        vid_frames.insert(0, vid_frames[0])
    for i in range(padding_frames - int(padding_frames / 2)):
        vid_frames.append(vid_frames[-1])
    return vid_frames


def create_batch(data_array, labels, verbose=False, prefix_path=DATASET_PATH):
    examples = []
    parsed_labels = []
    if verbose:
        lib.progress_bar(0, len(data_array), prefix="Created examples", suffix=f'Complete (0/{len(data_array)})',
                         length=30)
    for (index, (video_id, label)) in enumerate(data_array):
        examples.append(parse_video_frames_from_path(os.path.join(prefix_path, video_id)))
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


def get_labels():
    with open(LABELS_PATH, "r") as file:
        return list(map(lambda x: x.strip(), file.readlines()))


def get_data_ids_from_file(file):
    with open(file, "r") as file:
        csv_reader = csv.reader(file, delimiter=';')
        return [line for line in csv_reader]
