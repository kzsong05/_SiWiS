import json
import jsonlines
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor


def _calc_edge(bbox):
    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]

    edge = np.stack([width, height], axis=-1)
    return edge.astype(np.float32)


class BasicDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        num_antennas: int,
        video_frames: int,
        radio_frames: int,
        image_size: list,
        heatmap_size: list,
        heatmap_sigma: int,
        num_keypoint: int,
    ):
        self._load_data(data_file)
        self.num_antennas = num_antennas
        self.video_frames = video_frames
        self.radio_frames = radio_frames

        self.threadPool = ThreadPoolExecutor(max_workers=self.video_frames)
        self.heatmap_generator = HeatmapGenerator(
            image_size=image_size,
            heatmap_size=heatmap_size,
            heatmap_sigma=heatmap_sigma,
            num_keypoint=num_keypoint,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :return:
            video:      (str)
            mask:       (video_frames, image_size[1], image_size[0])
            hm:         (video_frames, keypoint, heatmap_size[1], heatmap_size[0])
            kp:         (video_frames, person, keypoint, 2)
            edge:       (person, 2)
            radio:      (video_frames, radio_frames, antennas_num, 2)
        """
        item_info = self.data[index]
        video = self._get_video(item_info["video"])
        mask = self._get_mask(item_info["mask"])
        hm, kp, edge = self._get_keypoint(item_info["keypoint"])
        radio = self._get_radio(item_info["radio"])

        return video, mask, hm, kp, edge, radio

    def _load_data(self, file_path):
        self.data = []
        if file_path:
            with jsonlines.open(file_path, mode='r') as rfile:
                for line in rfile:
                    self.data.append(line)

    def _get_video(self, video_clip):
        return video_clip[self.video_frames // 2]

    def _get_mask(self, files):
        mask = []
        threads = []

        def load_file(np_file):
            return np.load(np_file, allow_pickle=True)

        for file in files:
            thread = self.threadPool.submit(load_file, file)
            threads.append(thread)

        for thread in threads:
            mask.append(thread.result())
        mask = np.stack(mask, axis=0)

        return mask.astype(np.bool_)

    def _get_keypoint(self, files):
        heatmap = []
        keypoint = []
        edge = None
        threads = []

        def load_file(json_file):
            with open(json_file, 'r', encoding='utf-8') as rfile:
                anno = json.load(rfile)

            kp = np.array([item['keypoints'] for item in anno], dtype=np.float32)
            kp = np.delete(kp, [1, 2, 3, 4], axis=1)
            bbox = np.array([item['bbox'][0] for item in anno], dtype=np.float32)

            hm = self.heatmap_generator(kp)
            bbox = _calc_edge(bbox)
            return hm, kp, bbox

        for file in files:
            thread = self.threadPool.submit(load_file, file)
            threads.append(thread)

        for idx, thread in enumerate(threads):
            hm, kp, bbox = thread.result()
            heatmap.append(hm)
            keypoint.append(kp)
            if idx == self.video_frames // 2:
                edge = bbox

        heatmap = np.stack(heatmap, axis=0)
        return heatmap, keypoint, edge

    def _get_radio(self, files):
        radio = []
        threads = []

        def load_file(np_file):
            return np.load(np_file, allow_pickle=True)

        for file in files:
            thread = self.threadPool.submit(load_file, file)
            threads.append(thread)

        for thread in threads:
            radio.append(thread.result())
        radio = np.stack(radio, axis=0)

        return radio.astype(np.float32)


class HeatmapGenerator(object):
    def __init__(
        self,
        image_size: list,
        heatmap_size: list,
        heatmap_sigma: int,
        num_keypoint: int,
    ):
        super().__init__()

        self.image_size = np.array(image_size, dtype=np.int32)
        self.heatmap_size = np.array(heatmap_size, dtype=np.int32)
        self.feat_stride = self.image_size / self.heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.num_keypoint = num_keypoint

        self._generate_gaussian()

    def _generate_gaussian(self):
        self.radius_size = self.heatmap_sigma * 3
        size = 2 * self.radius_size + 1

        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

    def __call__(self, keypoint):
        heatmap = np.zeros((self.num_keypoint, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        for person_id in range(len(keypoint)):
            for keypoint_id in range(self.num_keypoint):
                mu_x = np.around(keypoint[person_id][keypoint_id][0] / self.feat_stride[0], decimals=0).astype(np.int32)
                mu_y = np.around(keypoint[person_id][keypoint_id][1] / self.feat_stride[1], decimals=0).astype(np.int32)

                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - self.radius_size, mu_y - self.radius_size]
                br = [mu_x + self.radius_size + 1, mu_y + self.radius_size + 1]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
                    continue

                # Usable gaussian range
                g_x = [max(0, -ul[0]), min(br[0], int(self.heatmap_size[0])) - ul[0]]
                g_y = [max(0, -ul[1]), min(br[1], int(self.heatmap_size[1])) - ul[1]]
                # Image range
                img_x = [max(0, ul[0]), min(br[0], int(self.heatmap_size[0]))]
                img_y = [max(0, ul[1]), min(br[1], int(self.heatmap_size[1]))]

                heatmap[keypoint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                    heatmap[keypoint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                    self.g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                )

        return heatmap.astype(np.float32)


