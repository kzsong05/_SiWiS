import os
import cv2
import numpy as np
from utils.check import check_visual_save_path, check_path


keypoint = [
    "Nose",
    "Chest",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle"
]


limbs = [
    [0, 1],  # 鼻子到胸部
    [1, 2],  # 胸部到左肩
    [2, 4],  # 左肩到左肘
    [4, 6],  # 左肘到左手腕
    [1, 3],  # 胸部到右肩
    [3, 5],  # 右肩到右肘
    [5, 7],  # 右肘到右手腕
    [1, 8],  # 胸部到左髋
    [1, 9],  # 胸部到右髋
    [8, 10],  # 左髋到左膝
    [10, 12],  # 左膝到左踝
    [9, 11],  # 右髋到右膝
    [11, 13]  # 右膝到右踝
]


colors = [
    (0, 69, 255),
    (0, 127, 255),
    (155, 211, 255),
    (139, 236, 255),
    (86, 114, 255),
    (186, 231, 255),
    (0, 238, 238),
    (148, 238, 78),
    (0, 238, 0),
    (238, 238, 0),
    (238, 104, 123),
    (255, 191, 0),
    (238, 121, 159)
]


class VisualKeypoint(object):
    def __init__(
        self,
        args,
    ):
        self.image_size = np.array(args.image_size, dtype=np.int32)
        self.heatmap_size = np.array(args.heatmap_size, dtype=np.int32)
        self.feat_stride = self.image_size / self.heatmap_size

        self.test_visual_sample = args.test_visual_sample
        self.visual_save_path = check_visual_save_path(args)

        self.draw_pred_keypoint_visual = args.draw_pred_keypoint_visual
        self.draw_gt_keypoint_visual = args.draw_gt_keypoint_visual
        self.draw_on_black_canvas = args.draw_on_black_canvas

    def eval_visual(self, samples, video, gt_kp, pred_kp, epoch):
        # video     (batch)
        # gt_kp     (batch, person, keypoint, 2)
        # pred_kp   (batch, person, keypoint, 3)
        for idx in samples:
            frame = self._draw_frame(video[idx], gt_kp[idx], pred_kp[idx])
            save_path = os.path.join(self.visual_save_path, f"checkpoint-{epoch}", "keypoint")
            check_path(save_path)
            cv2.imwrite(os.path.join(save_path, os.path.basename(video[idx])), frame)

    def test_visual(self, samples, video, gt_kp, pred_kp):
        # video     (batch)
        # gt_kp     (batch, person, keypoint, 2)
        # pred_kp   (batch, person, keypoint, 3)
        for idx in samples:
            frame = self._draw_frame(video[idx], gt_kp[idx], pred_kp[idx])
            save_path = os.path.join(self.visual_save_path, "test", "keypoint")
            check_path(save_path)
            cv2.imwrite(os.path.join(save_path, os.path.basename(video[idx])), frame)

    def _draw_frame(self, frame, gt_kp, pred_kp):
        frame = cv2.imread(frame)
        if self.draw_on_black_canvas:
            frame = np.zeros_like(frame, dtype=np.uint8)

        if self.draw_gt_keypoint_visual:
            frame = self._draw_gt_keypoint(frame, gt_kp)
        if self.draw_pred_keypoint_visual:
            pred_kp[:, :, :2] = (pred_kp[:, :, :2] * self.feat_stride + self.feat_stride / 2).astype(np.int32)
            frame = self._draw_pred_keypoint(frame, pred_kp)
        return frame

    def _draw_gt_keypoint(self, frame, gt_kp):
        tg_chest = np.mean(gt_kp[:, 1:3, :], axis=1, dtype=np.int32)
        gt_kp = np.array([np.insert(gt_kp[i], 1, tg_chest[i], axis=0) for i in range(len(gt_kp))], dtype=np.int32)

        for person in gt_kp:
            for idx in range(len(limbs)):
                pt1 = person[limbs[idx][0]]
                pt2 = person[limbs[idx][1]]
                if self.draw_pred_keypoint_visual:
                    frame = cv2.line(frame, tuple(pt1), tuple(pt2), color=(0, 255, 0), thickness=5)
                else:
                    frame = cv2.line(frame, tuple(pt1), tuple(pt2), color=colors[idx], thickness=5)
        return frame

    def _draw_pred_keypoint(self, frame, pred_kp):
        pred_chest = np.mean(pred_kp[:, 1:3, :], axis=1)
        pred_kp = np.array([np.insert(pred_kp[i], 1, pred_chest[i], axis=0) for i in range(len(pred_kp))])

        pred_conf = pred_kp[:, :, -1]
        pred_kp = pred_kp[:, :, :2].astype(np.int32)

        for person, conf in zip(pred_kp, pred_conf):
            for idx in range(len(limbs)):
                pt1 = person[limbs[idx][0], :2]
                pt1_conf = conf[limbs[idx][0]]
                pt2 = person[limbs[idx][1], :2]
                pt2_conf = conf[limbs[idx][1]]
                if self.draw_gt_keypoint_visual:
                    frame = _draw_pred_keypoint_line(frame, tuple(pt1), pt1_conf, tuple(pt2), pt2_conf, color=(0, 0, 255))
                else:
                    frame = _draw_pred_keypoint_line(frame, tuple(pt1), pt1_conf, tuple(pt2), pt2_conf, color=colors[idx])
        return frame


draw_keypoint_threshold = -1


def _draw_pred_keypoint_line(frame, pt1, pt1_conf, pt2, pt2_conf, color):
    if pt1_conf > draw_keypoint_threshold and pt2_conf > draw_keypoint_threshold:
        cv2.line(frame, pt1, pt2, color=color, thickness=5)
    elif pt1_conf > draw_keypoint_threshold:
        cv2.circle(frame, pt1, radius=5, color=color, thickness=-1)
    elif pt2_conf > draw_keypoint_threshold:
        cv2.circle(frame, pt2, radius=5, color=color, thickness=-1)
    return frame
