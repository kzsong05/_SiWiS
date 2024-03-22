import torch
import torchvision
from torch import nn
from einops import rearrange


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, pred, gt):
        loss = self.criterion(pred, gt)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, gt):
        pred = self.sigmoid(pred)
        smooth = 1.
        i_flat = pred.view(-1)
        t_flat = gt.view(-1)
        intersection = (i_flat * t_flat).sum()
        A_sum = torch.sum(i_flat * i_flat)
        B_sum = torch.sum(t_flat * t_flat)
        loss = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
        return loss


class MaskLoss(nn.Module):
    def __init__(
        self,
        bce_loss_weight: float,
        dice_loss_weight: float,
        heatmap_size: list,
        video_frames: int,
    ):
        super(MaskLoss, self).__init__()

        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.heatmap_size = heatmap_size
        self.video_frames = video_frames

        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, gt):
        # pred  (batch, video, height, width)
        # gt    (batch, video, height, width)
        
        gt = rearrange(gt, "N (V C) h w -> (N V) C h w", C=1)
        gt = torchvision.transforms.functional.resize(gt, size=(self.heatmap_size[1], self.heatmap_size[0]), antialias=False)
        gt = rearrange(gt, "(N V) C h w -> N (V C) h w", V=self.video_frames)
        gt = (gt > 0.5).float().detach()

        total_loss = self.bce_loss(pred, gt) * self.bce_loss_weight + self.dice_loss(pred, gt) * self.dice_loss_weight
        return total_loss
