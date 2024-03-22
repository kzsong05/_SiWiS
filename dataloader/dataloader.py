import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class BasicDataloader(object):
    def __init__(
        self,
        dataset,
        batch_size,
        workers_per_gpu
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers_per_gpu = workers_per_gpu

    def get_train_dataloader(self):
        distribute_sampler = DistributedSampler(self.dataset, shuffle=True)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.workers_per_gpu,
                                drop_last=False,
                                pin_memory=True,
                                sampler=distribute_sampler,
                                collate_fn=self._collect_func)
        return dataloader

    def get_test_dataloader(self):
        distribute_sampler = DistributedSampler(self.dataset, shuffle=False)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.workers_per_gpu,
                                drop_last=False,
                                pin_memory=True,
                                sampler=distribute_sampler,
                                collate_fn=self._collect_func)
        return dataloader

    def _collect_func(self, data):
        """
        :return:
            batch_video:    list,       [batch_size, str]
            batch_mask:     bool_,      ndarray(batch_size, video_frames, image_size[1], image_size[0])
            batch_hm:       float32,    ndarray(batch, video_frames, keypoint, heatmap_size[1], heatmap_size[0])
            batch_kp:       list,       [batch, video_frames, person, keypoint, 2]
            batch_edge:     list,       [batch, person, 2]
            batch_radio:    float32,    ndarray(batch_size, video_frames, radio_frames, num_antennas, 2)
        """
        batch_video = []
        batch_mask = []
        batch_hm = []
        batch_kp = []
        batch_edge = []
        batch_radio = []

        for video, mask, hm, kp, edge, radio in data:
            batch_video.append(video)
            batch_mask.append(mask)
            batch_hm.append(hm)
            batch_kp.append(kp)
            batch_edge.append(edge)
            batch_radio.append(radio)

        batch_mask = np.stack(batch_mask, axis=0)
        batch_hm = np.stack(batch_hm, axis=0)
        batch_radio = np.stack(batch_radio, axis=0)

        return batch_video, batch_mask, batch_hm, batch_kp, batch_edge, batch_radio
