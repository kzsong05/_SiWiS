import os
import shutil
import jsonlines
import numpy as np
from tqdm import tqdm
from utils.basic import get_path_list, get_radio_list, get_video_list


class DatasetGenerator(object):
    def __init__(
        self,
        antennas_num: int,
        video_frames: int,
        radio_frames: int,
        video_frames_gap: int,
    ):
        super(DatasetGenerator, self).__init__()

        self.antennas_num = antennas_num
        self.video_frames = video_frames
        self.radio_frames = radio_frames
        self.video_frames_gap = video_frames_gap

    def _radio_and_match(self, path, radio_list, video_list):
        radio_generator = RadioGenerator(
            radio_frames=self.radio_frames,
            path=path,
            radio_list=radio_list,
            video_list=video_list,
        )
        radio_generator.generate()

        match_generator = MatchGenerator(
            video_frames=self.video_frames,
            video_frames_gap=self.video_frames_gap,
            path=path,
            video_list=video_list
        )
        match_generator.generate()

    def generate(self, path_list=None):
        path_list = get_path_list(path_list)

        for path in path_list:
            radio_list = get_radio_list(path, self.antennas_num)
            video_list = get_video_list(path)

            if os.path.exists(os.path.join(path, 'dataset.jsonl')):
                os.remove(os.path.join(path, 'dataset.jsonl'))

            if os.path.exists(os.path.join(path, "radio")):
                shutil.rmtree(os.path.join(path, 'radio'), ignore_errors=True)
            os.mkdir(os.path.join(path, 'radio'))

            self._radio_and_match(path, radio_list, video_list)


class RadioGenerator(object):
    def __init__(
        self,
        radio_frames: int,
        path: str,
        radio_list: list,
        video_list: list,
    ):
        super(RadioGenerator, self).__init__()

        self.radio_frames = radio_frames
        self.path = path
        self.radio_list = radio_list
        self.video_list = video_list
        self.video2radio = self._timestamp_match()

    def _timestamp_match(self):
        video2radio = dict()
        radio_frame_idx = 0
        for time, frame in self.video_list:
            while radio_frame_idx < len(self.radio_list) and self.radio_list[radio_frame_idx][0] <= time:
                radio_frame_idx += 1

            if 0 < radio_frame_idx < len(self.radio_list):
                video2radio[time] = radio_frame_idx - 1
            else:
                video2radio[time] = None

        return video2radio

    def _check_radios(self, time):
        if self.video2radio[time] is None or self.video2radio[time] < self.radio_frames - 1:
            return False
        return True

    def _generate_radio_chunk(self, time):
        radio_frame_idx = self.video2radio[time]
        radio_chunk = self.radio_list[radio_frame_idx - (self.radio_frames - 1): radio_frame_idx + 1]
        radio_chunk = np.stack([radio[1] for radio in radio_chunk], axis=0)
        return radio_chunk

    def generate(self):
        for time, frame in tqdm(self.video_list, desc=f"Generating Radios {self.path}: "):
            if self._check_radios(time):
                radio_chunk = self._generate_radio_chunk(time)
                radio_file = os.path.join(self.path, "radio", f"{time}.npy")
                np.save(radio_file, radio_chunk, allow_pickle=True)


class MatchGenerator(object):
    def __init__(
        self,
        video_frames: int,
        video_frames_gap: int,
        path: str,
        video_list: list,
    ):
        super(MatchGenerator, self).__init__()

        self.video_frames = video_frames
        self.video_frames_gap = video_frames_gap

        self.path = path
        self.video_list = video_list

    def _generate_video_clip(self, mid_idx):
        video_clip = []

        start = mid_idx - (self.video_frames // 2 * self.video_frames_gap)
        end = mid_idx + (self.video_frames // 2 * self.video_frames_gap) + 1
        for frame_idx in range(start, end, self.video_frames_gap):
            if frame_idx < 0 or frame_idx >= len(self.video_list):
                return None
            video_clip.append(self.video_list[frame_idx])
        return video_clip

    def _check_radios(self, video_clip):
        for time, frame in video_clip:
            radio_file = os.path.join(self.path, "radio", f"{time}.npy")
            if not os.path.exists(radio_file):
                return False
        return True

    def _write_data(self, video_clip):
        video = []
        mask = []
        keypoint = []
        radio = []
        for time, frame in video_clip:
            video.append(frame)
            mask.append(os.path.join(self.path, "mask", "heatmap", f"{time}.npy"))
            keypoint.append(os.path.join(self.path, "keypoint", f"{time}.json"))
            radio.append(os.path.join(self.path, "radio", f"{time}.npy"))

        with jsonlines.open(os.path.join(self.path, 'dataset.jsonl'), mode='a') as wfile:
            wfile.write({"video": video, "mask": mask, "keypoint": keypoint, "radio": radio})

    def generate(self):
        for index in tqdm(range(len(self.video_list)), desc=f"Match Dataset {self.path}: "):
            video_clip = self._generate_video_clip(index)
            if video_clip and self._check_radios(video_clip):
                self._write_data(video_clip)
