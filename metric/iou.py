import torch
import torchvision
from torchmetrics import Metric


def _calc_iou(pred, gt):
    smooth = 1.
    intersection = torch.sum(pred & gt, dim=(1, 2))
    union = torch.sum(pred | gt, dim=(1, 2))
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score


class IOUMetric(Metric):
    def __init__(
        self,
        heatmap_size: list,
        pred_mask_threshold: float,
    ):
        super(IOUMetric, self).__init__()

        self.heatmap_size = heatmap_size
        self.pred_mask_threshold = pred_mask_threshold
        self.iou_thresh = torch.linspace(0.5, 0.95, steps=int(((0.95 - 0.5) / 0.05) + 1))

        self.add_state("iou_count", default=torch.zeros((len(self.iou_thresh),)), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.zeros((1,)), dist_reduce_fx="sum")

    def update(self, pred, gt):
        # pred:     float32,    (batch, height, width)
        # gt:       bool_,      (batch, height, width)
        pred = pred > self.pred_mask_threshold

        gt = torchvision.transforms.functional.resize(gt.unsqueeze(1), size=(self.heatmap_size[1], self.heatmap_size[0]), antialias=False)
        gt = gt.squeeze(1) > 0.5

        iou_score = _calc_iou(pred, gt)
        for idx, thresh in enumerate(self.iou_thresh):
            self.iou_count[idx] += torch.sum(iou_score >= thresh)
        self.total_count += len(iou_score)

    def compute(self):
        ap = torch.mean(self.iou_count / self.total_count)
        ap_50 = self.iou_count[0] / self.total_count
        ap_60 = self.iou_count[2] / self.total_count
        ap_70 = self.iou_count[4] / self.total_count
        ap_80 = self.iou_count[6] / self.total_count
        return ap, ap_50, ap_60, ap_70, ap_80
