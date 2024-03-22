import torch
from torch import nn
import numpy as np


class MSELoss(nn.Module):
    def __init__(
        self,
        mse_loss_weight: float,
    ):
        super(MSELoss, self).__init__()
        self.mse_loss_weight = mse_loss_weight
        # self.criterion = nn.MSELoss(reduction="mean")
        self.keypoint_weight = torch.tensor([1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5], dtype=torch.float16)

    def forward(self, pred, gt):
        # pred      (batch, video_frames, keypoint, height, width)
        # gt        (batch, video_frames, keypoint, height, width)
        assert pred.shape == gt.shape

        loss = 0.
        for kp_idx in range(len(self.keypoint_weight)):
            weight = gt[:, :, kp_idx, :, :] + 1
            loss += torch.mean(weight * (pred[:, :, kp_idx, :, :] - gt[:, :, kp_idx, :, :]) ** 2 * self.keypoint_weight[kp_idx])
            # loss += self.criterion(pred[:, :, kp_idx, :, :], gt[:, :, kp_idx, :, :]) * self.keypoint_weight[kp_idx]
        return (loss / len(self.keypoint_weight)) * self.mse_loss_weight


class GroupLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        within_loss_weight: float,
        across_loss_weight: float,
        image_size: list,
        heatmap_size: list,
    ):
        super(GroupLoss, self).__init__()
        self.device = device
        self.within_loss_weight = within_loss_weight
        self.across_loss_weight = across_loss_weight

        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.feat_stride = self.image_size / self.heatmap_size

    def _calc_within_loss(self, pred, gt):
        within_loss = torch.zeros(1, dtype=torch.float32, device=self.device)

        reference_embeds = []
        for person_idx, person in enumerate(gt):
            keypoint_embed = []
            for kp_idx, kp in enumerate(person):
                if 0 <= kp[0] < self.heatmap_size[0] and 0 <= kp[1] < self.heatmap_size[1]:
                    keypoint_embed.append(pred[kp_idx, kp[1], kp[0]])
            if len(keypoint_embed) == 0:
                continue
            keypoint_embed = torch.stack(keypoint_embed, dim=0)
            person_embed = torch.mean(keypoint_embed, dim=0)

            within_loss += torch.mean((keypoint_embed - person_embed.expand_as(keypoint_embed)) ** 2)
            reference_embeds.append(person_embed)

        reference_embeds = torch.stack(reference_embeds, dim=0)
        num_person = len(gt)
        return within_loss / num_person, reference_embeds

    def _calc_across_loss(self, reference_embeds):
        num_person = reference_embeds.shape[0]
        reference_expand = reference_embeds.expand(num_person, num_person)
        reference_diff = reference_expand - reference_expand.transpose(0, 1)

        reference_mask = 1 - torch.eye(num_person).to(reference_embeds.device)
        reference_diff = (1 - torch.abs(reference_diff)) * reference_mask
        across_loss = torch.clamp(reference_diff, min=0).sum()
        return across_loss / reference_mask.sum()

    def _calc_loss(self, pred, gt):
        gt = np.around(gt / self.feat_stride, decimals=0).astype(np.int32)

        within_loss, reference_embeds = self._calc_within_loss(pred, gt)
        if len(gt) <= 1:
            return within_loss * self.within_loss_weight, torch.zeros(1, dtype=torch.float32, device=self.device)

        across_loss = self._calc_across_loss(reference_embeds)
        return within_loss * self.within_loss_weight, across_loss * self.across_loss_weight

    def forward(self, preds, gt):
        # pred  (batch, video_frames, keypoint, height, width)
        # gt    (batch, video_frames, num_person, keypoint, 2)
        total_within_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        total_across_loss = torch.zeros(1, dtype=torch.float32, device=self.device)

        preds = preds.flatten(0, 1)
        gt = [v for b in gt for v in b]

        for pred, kp in zip(preds, gt):
            within_loss, across_loss = self._calc_loss(pred, kp)
            total_within_loss += within_loss
            total_across_loss += across_loss
        return total_within_loss / len(preds), total_across_loss / len(preds)


class KeypointLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        mse_loss_weight: float,
        within_loss_weight: float,
        across_loss_weight: float,
        image_size: list,
        heatmap_size: list
    ):
        super(KeypointLoss, self).__init__()

        self.mse_loss = MSELoss(
            mse_loss_weight=mse_loss_weight,
        )
        self.group_loss = GroupLoss(
            device=device,
            within_loss_weight=within_loss_weight,
            across_loss_weight=across_loss_weight,
            image_size=image_size,
            heatmap_size=heatmap_size,
        )

    def forward(self, pred_hm, pred_tag, hm, kp):
        # pred_hm   (batch, video_frames, keypoint, height, width)
        # pred_tag  (batch, video_frames, keypoint, height, width)
        # hm        (batch, video_frames, keypoint, height, width)
        # kp        (batch, video_frames, num_person, keypoint, 2)
        mse_loss = self.mse_loss(pred_hm, hm)
        within_loss, across_loss = self.group_loss(pred_tag, kp)

        return mse_loss, within_loss, across_loss
