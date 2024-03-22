import os
import random
import jsonlines
from utils.basic import get_path_list


class DatasetSplit(object):
    def __init__(
        self,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2
    ):
        super().__init__()

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.train_file = "train_data.jsonl"
        self.test_file = "test_data.jsonl"

    def _load_dataset(self, data_path):
        dataset_file = os.path.join(data_path, "dataset.jsonl")

        total_lines = []
        with jsonlines.open(dataset_file, mode='r') as rfile:
            for line in rfile:
                total_lines.append(line)
        return total_lines

    def _write_data(self, lines, file):
        with jsonlines.open(file, mode='a') as wfile:
            for line in lines:
                video_path = [os.path.join("data", frame) for frame in line["video"]]
                mask_path = [os.path.join("data", mask) for mask in line["mask"]]
                keypoint_path = [os.path.join("data", keypoint) for keypoint in line['keypoint']]
                radio_path = [os.path.join("data", radio) for radio in line["radio"]]

                wfile.write({"video": video_path, "mask": mask_path, "keypoint": keypoint_path, "radio": radio_path})

    def process(self, path_lists=None):
        for file in [self.train_file, self.test_file]:
            if os.path.exists(file):
                os.remove(file)
        path_lists = get_path_list(path_lists)

        train_total = 0
        test_total = 0

        for path in path_lists:
            lines = self._load_dataset(path)
            train_length = int(self.train_ratio * len(lines))

            train_data = lines[:train_length]
            test_data = lines[train_length:]

            self._write_data(train_data, self.train_file)
            self._write_data(test_data, self.test_file)

            print(f"Data Split {path}: All {len(lines)}; Train {train_length}; Test {len(lines) - train_length}")

            train_total += train_length
            test_total += len(lines) - train_length

        print(f"Total Count: Train {train_total}; Test{test_total}")
