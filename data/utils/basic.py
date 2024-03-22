import os
import numpy as np


def get_path_list(path_list=None):
    if path_list:
        path_list = [os.path.join("raw", path) for path in path_list]
    else:
        path_list = [os.path.join("raw", path) for path in os.listdir("raw")]
    return path_list


def get_radio_list(path, antennas_num):
    radio_list = list()

    with open(os.path.join(path, "radio.txt"), "r") as file:
        for line in file.readlines():
            tokens = [token for token in line.split() if token]
            if len(tokens) != (antennas_num * 2 + 1):
                break
            timestamp = int(tokens[0])
            radios = np.array(tokens[1:], dtype=np.float32).reshape(antennas_num, 2)

            radio_list.append((timestamp, radios))

    def get_element(element):
        return element[0]
    radio_list.sort(key=get_element, reverse=False)

    return radio_list


def get_video_list(path):
    video_list = list()

    files = os.listdir(os.path.join(path, "video"))
    for file in files:
        video_list.append((int(file.split('.')[0]), os.path.join(path, "video", file)))

    def get_element(element):
        return element[0]
    video_list.sort(key=get_element, reverse=False)

    return video_list
