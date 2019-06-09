import numpy as np
import os
import csv
import json

DATASET_PATH = "D:/Datasets/20bn-jester/extracted"
TRAIN_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-train.csv"


if __name__ == "__main__":
    video_id = "88829"
    print(len(os.listdir(os.path.join(DATASET_PATH, video_id))))

    for (index, frame) in enumerate(os.listdir(os.path.join(DATASET_PATH, video_id))):
        new_name = f"{'{:05d}'.format(int(frame[:frame.index('.')]) - 5)}.jpg"
        os.rename(os.path.join(DATASET_PATH, video_id, frame), os.path.join(DATASET_PATH, video_id, new_name))
        print(index, frame, new_name)