import numpy as np
import os
import csv
import json
import lib

DATASET_PATH = "D:/Datasets/20bn-jester/extracted"
TRAIN_LIST_PATH = "D:/Datasets/20bn-jester/jester-v1-validation.csv"


if __name__ == "__main__":
    with open(TRAIN_LIST_PATH, "r") as file:
        csv_reader = csv.reader(file, delimiter=';')
        exes = [line for line in csv_reader]
    scoreboard = {}
    index = 0
    videos = len(exes)
    big_videos = []

    lib.progress_bar(0, videos, prefix="Checking video:", suffix=f"Completed (0/{videos})", length=40)

    for (index, (video_id, _)) in enumerate(exes):
        frame_count = len(os.listdir(os.path.join(DATASET_PATH, video_id)))

        if frame_count > 45:
            big_videos.append(video_id)
        if scoreboard.get(str(frame_count)):
            scoreboard[str(frame_count)] += 1
        else:
            scoreboard[str(frame_count)] = 1
        index += 1
        lib.progress_bar(index + 1, videos, prefix=" Checking video:", suffix=f"Completed ({index + 1}/{videos})", length=40)

    print(scoreboard)
    print(big_videos)

    # with open('data.json', 'w') as f:  # writing JSON object
    #    json.dump({
    #        'big_vids': big_videos,
    #        'scores': scoreboard
    #    }, f)
