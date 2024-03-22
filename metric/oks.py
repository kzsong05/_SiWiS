import torch
import numpy as np
from munkres import Munkres
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class OKSMetric(Metric):
    def __init__(
        self,
        device,
        num_keypoint: int,
        image_size: list,
        heatmap_size: list,
    ):
        super(OKSMetric, self).__init__()

        self.cuda_device = device
        self.num_keypoint = num_keypoint
        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.feat_stride = self.image_size / self.heatmap_size

        self.oks_thresh = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recall_thresh = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.sigmas = np.array([.26, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

        self.munkres = Munkres()

        self.add_state("pred_match", default=[], dist_reduce_fx="cat")
        self.add_state("pred_conf", default=[], dist_reduce_fx="cat")
        self.add_state("gt_count", default=torch.zeros((1,)), dist_reduce_fx="sum")

    def update(self, pred_kp, pred_conf, gt_kp, gt_edge):
        # pred_kp       (person, keypoint, 3)
        # pred_conf     (person)
        # gt_kp         (person, keypoint, 2)
        # gt_edge       (person, 2)

        gt_kp = np.around(gt_kp / self.feat_stride, decimals=0).astype(np.int32)
        gt_edge = np.around(gt_edge / self.feat_stride, decimals=0).astype(np.int32)

        oks_score = self._calc_oks(pred_kp[:, :, :2], gt_kp, gt_edge)
        if oks_score is not None:
            pred_match, gt_match = self._person_match(oks_score, pred_conf)
            self.pred_match.append(torch.from_numpy(pred_match).transpose(0, 1).to(self.cuda_device))
            self.pred_conf.append(torch.from_numpy(pred_conf).to(self.cuda_device))
            self.gt_count += gt_match.shape[1]

    def _calc_oks(self, pred, gt, edge):
        pred_num = pred.shape[0]
        gt_num = gt.shape[0]
        if pred_num == 0 or gt_num == 0:
            return None

        areas = edge[:, 0] * edge[:, 1]
        oks_score = np.zeros((pred_num, gt_num), dtype=np.float32)
        for j in range(gt_num):
            for i in range(pred_num):
                e = np.sum((pred[i] - gt[j]) ** 2, axis=-1) / (2 * (self.sigmas ** 2) * areas[j])
                oks_score[i, j] = np.sum(np.exp(-e)) / self.num_keypoint
        return oks_score

    def _munkres_match(self, oks, conf, thresh):
        pred_num, gt_num = oks.shape
        conf = conf.reshape(-1, 1)

        if pred_num > gt_num:
            oks = np.concatenate((oks, np.zeros((pred_num, pred_num - gt_num))), axis=1)
        elif pred_num < gt_num:
            oks = np.concatenate((oks, np.zeros((gt_num - pred_num, gt_num))), axis=0)
            conf = np.concatenate((conf, np.zeros(gt_num - pred_num).reshape(-1, 1)))

        pairs = self.munkres.compute(np.where(oks >= thresh, -conf, 1e6))

        match_result = []
        for row, col in pairs:
            if 0 <= row < pred_num and 0 <= col < gt_num and oks[row][col] >= thresh:
                match_result.append((row, col))
        return match_result

    def _person_match(self, oks, conf):
        pred_match = np.zeros((len(self.oks_thresh), oks.shape[0])).astype(np.bool_)
        gt_match = np.zeros((len(self.oks_thresh), oks.shape[1])).astype(np.bool_)

        for idx, t in enumerate(self.oks_thresh):
            pairs = self._munkres_match(oks, conf, t)
            for row, col in pairs:
                pred_match[idx, row] = True
                gt_match[idx, col] = True

        return pred_match, gt_match

    def compute(self):
        pred_match = dim_zero_cat(self.pred_match).transpose(0, 1).cpu().numpy()
        pred_conf = dim_zero_cat(self.pred_conf).cpu().numpy()
        gt_count = self.gt_count.cpu().numpy()

        precision_result = np.zeros((len(self.oks_thresh), len(self.recall_thresh)))
        recall_result = np.zeros((len(self.oks_thresh)))

        pred_conf_idx = np.argsort(-pred_conf)
        pred_match = pred_match[:, pred_conf_idx]

        tp_sum = np.cumsum(pred_match, axis=1).astype(np.float32)
        fp_sum = np.cumsum(~pred_match, axis=1).astype(np.float32)

        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            tp, fp = np.array(tp), np.array(fp)
            recall = tp / gt_count
            precision = tp / (tp + fp + np.spacing(1))
            q = np.zeros((len(self.recall_thresh)))

            recall_result[t] = recall[-1]

            precision, q = precision.tolist(), q.tolist()
            for i in range(len(precision) - 1, 0, -1):
                if precision[i] > precision[i - 1]:
                    precision[i - 1] = precision[i]

            indexes = np.searchsorted(recall, self.recall_thresh, side="left")
            try:
                for ri, pi in enumerate(indexes):
                    q[ri] = precision[pi]
            except:
                pass
            precision_result[t, :] = np.array(q)

        ap = np.mean(precision_result)
        ap_50 = np.mean(precision_result[np.where(self.oks_thresh == 0.5)[0][0], :])
        ap_60 = np.mean(precision_result[np.where(self.oks_thresh == 0.6)[0][0], :])
        ap_70 = np.mean(precision_result[np.where(self.oks_thresh == 0.7)[0][0], :])
        ap_80 = np.mean(precision_result[np.where(self.oks_thresh == 0.8)[0][0], :])
        ar = np.mean(recall_result)
        return ap, ap_50, ap_60, ap_70, ap_80, ar
