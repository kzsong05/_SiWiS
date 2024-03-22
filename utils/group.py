import torch
import torch.nn as nn
import numpy as np
from munkres import Munkres


class KeypointGroup(object):
    def __init__(
        self,
        num_keypoint: int,
        heatmap_size: list,
    ):
        super(KeypointGroup, self).__init__()

        self.num_keypoint = num_keypoint
        self.heatmap_size = heatmap_size

        self.max_num_person = 10
        self.group_heatmap_threshold = 0.2
        self.group_tag_threshold = 1
        self.group_confidence_threshold = 8

        self.munkres = Munkres()
        self.keypoint_group_order = [0, 1, 2, 7, 8, 3, 4, 5, 6, 9, 10, 11, 12]
        assert len(self.keypoint_group_order) == self.num_keypoint

    def _non_maximum_suppression(self, heatmap):
        maximum = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(heatmap)
        mask = torch.eq(maximum, heatmap).float()
        heatmap = heatmap * mask
        return heatmap

    def _top_k(self, heatmap, tag):
        heatmap, tag = torch.from_numpy(heatmap), torch.from_numpy(tag)
        assert heatmap.shape == tag.shape

        heatmap = self._non_maximum_suppression(heatmap)
        heatmap = torch.flatten(heatmap, -2, -1)
        val_k, index_k = torch.topk(heatmap, k=self.max_num_person, dim=-1)

        tags = torch.flatten(tag, -2, -1)
        tag_k = torch.gather(tags, dim=-1, index=index_k)

        index_k = torch.stack([index_k % self.heatmap_size[0], index_k // self.heatmap_size[0]], dim=-1)
        return val_k.numpy(), index_k.numpy(), tag_k.numpy()

    def _group_by_tag(self, vals, indexes, tags):
        grouped_keypoint = dict()
        for keypoint_idx in self.keypoint_group_order:
            mask = vals[keypoint_idx, :] > self.group_heatmap_threshold
            if mask.sum() <= 0:
                continue

            val_k = vals[keypoint_idx, :][mask]
            index_k = indexes[keypoint_idx, :][mask]
            tag_k = tags[keypoint_idx, :][mask]

            if not grouped_keypoint:
                for val, index, tag in zip(val_k, index_k, tag_k):
                    key = len(grouped_keypoint.keys())
                    grouped_keypoint[key] = [[0, 0, 0, 0] for _ in range(self.num_keypoint)]
                    grouped_keypoint[key][keypoint_idx] = [index[0], index[1], val, tag]
            else:
                grouped_keys = list(grouped_keypoint.keys())
                grouped_tags = [np.mean([keypoint[3] for keypoint in grouped_keypoint[key] if keypoint[2] > 0]) for key in grouped_keys]

                tags_diff = np.abs(tag_k.reshape(-1, 1) - np.array(grouped_tags))
                tags_diff_weight = np.round(tags_diff) * 100 - val_k.reshape(-1, 1)
                diff_h, diff_w = tags_diff_weight.shape
                if diff_h > diff_w:
                    tags_diff_weight = np.concatenate([tags_diff_weight, np.zeros((diff_h, diff_h - diff_w)) + 1e10], axis=1)

                pairs = np.array(self.munkres.compute(tags_diff_weight)).astype(np.int32)
                for row, col in pairs:
                    if row < diff_h and col < diff_w and tags_diff[row][col] < self.group_tag_threshold:
                        key = grouped_keys[col]
                        grouped_keypoint[key][keypoint_idx] = [index_k[row][0], index_k[row][1], val_k[row], tag_k[row]]
                    else:
                        key = len(grouped_keypoint.keys())
                        grouped_keypoint[key] = [[0, 0, 0, 0] for _ in range(self.num_keypoint)]
                        grouped_keypoint[key][keypoint_idx] = [index_k[row][0], index_k[row][1], val_k[row], tag_k[row]]

        keypoint_group = [grouped_keypoint[key] for key in grouped_keypoint.keys()]
        return keypoint_group

    def _group_conf_filter(self, group):
        filtered_group = []
        for person in group:
            conf = person[:, 2]
            num_count = np.sum(conf > self.group_heatmap_threshold)
            if num_count > self.group_confidence_threshold:
                filtered_group.append(person)
        return np.array(filtered_group)

    def _keypoint_refine(self, heatmap, tag, group):
        person_tag = np.mean([keypoint[3] for keypoint in group if keypoint[2] > 0])

        for keypoint_idx in range(self.num_keypoint):
            if group[keypoint_idx][2] == 0:
                val = heatmap[keypoint_idx, :, :]
                tag_dist = np.abs(tag[keypoint_idx, :, :] - person_tag)
                keypoint_score = val - np.round(tag_dist)
                h, w = np.unravel_index(np.argmax(keypoint_score.flatten()), keypoint_score.shape)
                group[keypoint_idx] = [w, h, heatmap[keypoint_idx, h, w], tag[keypoint_idx, h, w]]
        return group

    def group(self, heatmap, tag):
        # heatmap   (keypoint, height, width)
        # tag       (keypoint, height, width)
        val_k, index_k, tag_k = self._top_k(heatmap, tag)
        keypoint_group = np.array(self._group_by_tag(val_k, index_k, tag_k))  # (person, keypoint, 4)
        keypoint_group = self._group_conf_filter(keypoint_group)
        group_refine = np.array([self._keypoint_refine(heatmap, tag, person) for person in keypoint_group]) if len(keypoint_group) else np.array([])

        group = group_refine[:, :, :3] if len(group_refine) else np.zeros((1, 13, 3))
        conf = np.mean(group_refine[:, :, 2], axis=1) if len(group_refine) else np.zeros((1,))
        return group, conf
