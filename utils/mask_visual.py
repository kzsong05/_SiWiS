import os
import cv2
import numpy as np
from utils.check import check_visual_save_path, check_path


class VisualMask(object):
    def __init__(
        self,
        args,
    ):
        self.image_size = np.array(args.image_size, dtype=np.int32)
        self.heatmap_size = np.array(args.heatmap_size, dtype=np.int32)

        self.test_visual_sample = args.test_visual_sample
        self.visual_save_path = check_visual_save_path(args)

        self.pred_mask_threshold = args.pred_mask_threshold
        self.draw_pred_mask_visual = args.draw_pred_mask_visual
        self.draw_gt_mask_visual = args.draw_gt_mask_visual
        self.draw_on_black_canvas = args.draw_on_black_canvas

    def eval_visual(self, samples, video, gt, pred, epoch):
        # video     (batch)
        # pred      (batch, height, width)
        # gt        (batch, height, width)
        for idx in samples:
            frame = self._draw_frame(video[idx], gt[idx], pred[idx])
            save_path = os.path.join(self.visual_save_path, f"checkpoint-{epoch}", "mask")
            check_path(save_path)
            cv2.imwrite(os.path.join(save_path, os.path.basename(video[idx])), frame)

    def test_visual(self, samples, video, gt, pred):
        # video     (batch)
        # pred      (batch, height, width)
        # gt        (batch, height, width)
        for idx in samples:
            frame = self._draw_frame(video[idx], gt[idx], pred[idx])
            save_path = os.path.join(self.visual_save_path, "test", "mask")
            check_path(save_path)
            cv2.imwrite(os.path.join(save_path, os.path.basename(video[idx])), frame)

    def _draw_frame(self, frame, gt, pred):
        frame = cv2.imread(frame)
        if self.draw_on_black_canvas:
            frame = np.zeros_like(frame, dtype=np.uint8)

        if self.draw_gt_mask_visual:
            frame = self._draw_gt_mask(frame, gt)
        if self.draw_pred_mask_visual:
            pred = cv2.resize(pred, self.image_size)
            pred = pred > self.pred_mask_threshold
            frame = self._draw_pred_mask(frame, pred)
        return frame

    def _draw_pred_mask(self, frame, pred):
        assert pred.dtype == np.bool_

        mask = np.zeros_like(frame)
        if self.draw_on_black_canvas:
            mask[pred, :] = [255, 255, 255]
            frame = cv2.addWeighted(frame, 1, mask, 1, 0)
        else:
            mask[pred, :] = [0, 0, 255]  # Red color
            frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)
        return frame

    def _draw_gt_mask(self, frame, gt):
        assert gt.dtype == np.bool_

        mask = np.zeros_like(frame)
        if self.draw_on_black_canvas:
            mask[gt, :] = [255, 255, 255]
            frame = cv2.addWeighted(frame, 1, mask, 1, 0)
        else:
            mask[gt, :] = [0, 255, 0]  # Green color
            frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)
        return frame
