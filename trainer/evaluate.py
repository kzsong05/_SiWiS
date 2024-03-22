import torch
import wandb
import random
import numpy as np
from tqdm import tqdm
from utils.basic import Logger
from utils.mask_visual import VisualMask
from utils.group import KeypointGroup
from utils.keypoint_visual import VisualKeypoint


class Evaluator(object):
    def __init__(
        self,
        args,
        model,
        mask_metric,
        keypoint_metric,
        test_dataset,
        test_dataloader
    ):

        self.args = args
        self.device = args.device
        self.logger = Logger(__name__).get_logger(args.log_file)

        self.model = model
        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader

        self.mask_metric = mask_metric
        self.mask_visualizer = VisualMask(args)

        self.keypoint_group = KeypointGroup(
            num_keypoint=args.num_keypoint,
            heatmap_size=args.heatmap_size,
        )
        self.keypoint_metric = keypoint_metric
        self.keypoint_visualizer = VisualKeypoint(args)
        self.samples = None

    @torch.no_grad()
    def eval(self, epoch):
        self.model.eval()

        if self.args.local_rank == 0:
            self.logger.info(f"***** Run eval *****")
        batch_videos = []       # (batch)
        batch_gt_mask = []      # (batch, height, width)
        batch_pred_mask = []    # (batch, height, width)
        batch_gt_kp = []        # (batch, person, keypoint, 2)
        batch_pred_kp = []      # (batch, person, keypoint, 3)

        for video, mask, hm, kp, edge, radio in self.test_dataloader:
            tensors = [mask, hm, radio]
            mask, hm, radio = [torch.from_numpy(tensor).to(self.device) for tensor in tensors]

            pred_mask, pred_hm, pred_tag = self.model(radio)

            pred_mask = pred_mask[:, self.args.video_frames // 2]
            mask = mask[:, self.args.video_frames // 2]
            self.mask_metric.update(pred_mask, mask)

            pred_hm = pred_hm[:, self.args.video_frames // 2].cpu().numpy()
            pred_tag = pred_tag[:, self.args.video_frames // 2].cpu().numpy()
            kp = [frames[self.args.video_frames // 2] for frames in kp]

            pred_kp = []
            for per_hm, per_tag, per_kp, per_edge in zip(pred_hm, pred_tag, kp, edge):
                group, conf = self.keypoint_group.group(per_hm, per_tag)
                self.keypoint_metric.update(group, conf, per_kp, per_edge)
                pred_kp.append(group)

            batch_videos.extend(video)
            batch_gt_mask.append(mask.cpu().numpy())
            batch_pred_mask.append(pred_mask.cpu().numpy())
            batch_gt_kp.extend(kp)
            batch_pred_kp.extend(pred_kp)

        mask_ap, mask_ap_50, mask_ap_60, mask_ap_70, mask_ap_80 = self.mask_metric.compute()
        kp_ap, kp_ap_50, kp_ap_60, kp_ap_70, kp_ap_80, kp_ar = self.keypoint_metric.compute()
        if self.args.local_rank == 0:
            if self.args.wandb_enable:
                wandb.log({
                    "eval/mask AP": mask_ap,
                    "eval/mask AP@.5": mask_ap_50,
                    "eval/mask AP@.6": mask_ap_60,
                    "eval/mask AP@.7": mask_ap_70,
                    "eval/mask AP@.8": mask_ap_80,
                    "eval/keypoint AP": kp_ap,
                    "eval/keypoint AP@.5": kp_ap_50,
                    "eval/keypoint AP@.6": kp_ap_60,
                    "eval/keypoint AP@.7": kp_ap_70,
                    "eval/keypoint AP@.8": kp_ap_80,
                    "eval/keypoint AR": kp_ar,
                    "train/epoch": epoch,
                })
            self.logger.info(
                f"Epoch {epoch} Eval Result:\n"
                f"---Mask--- AP: {mask_ap * 100:.2f}, AP@.5: {mask_ap_50 * 100:.2f}, AP@.6: {mask_ap_60 * 100:.2f}, AP@.7: {mask_ap_70 * 100:.2f}, AP@.8: {mask_ap_80 * 100:.2f}\n"
                f"---Keypoint--- AP: {kp_ap * 100:.2f}, AP@.5: {kp_ap_50 * 100:.2f}, AP@.6: {kp_ap_60 * 100:.2f}, AP@.7: {kp_ap_70 * 100:.2f}, AP@.8: {kp_ap_80 * 100:.2f}, AR: {kp_ar * 100:.2f}")

        self.mask_metric.reset()
        self.keypoint_metric.reset()

        if self.args.do_eval_visual:
            batch_gt_mask = np.concatenate(batch_gt_mask, axis=0)
            batch_pred_mask = np.concatenate(batch_pred_mask, axis=0)

            if self.samples is None:
                num_sample = min(self.args.eval_visual_sample, len(batch_videos)) if self.args.eval_visual_sample > 0 else len(batch_videos)
                self.samples = random.sample(range(len(batch_videos)), num_sample)

            self.mask_visualizer.eval_visual(self.samples, batch_videos, batch_gt_mask, batch_pred_mask, epoch)
            self.keypoint_visualizer.eval_visual(self.samples, batch_videos, batch_gt_kp, batch_pred_kp, epoch)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        if self.args.local_rank == 0:
            self.logger.info(f"***** Run test *****")
        batch_videos = []       # (batch)
        batch_gt_mask = []      # (batch, height, width)
        batch_pred_mask = []    # (batch, height, width)
        batch_gt_kp = []        # (batch, person, keypoint, 2)
        batch_pred_kp = []      # (batch, person, keypoint, 3)

        for video, mask, hm, kp, edge, radio in tqdm(self.test_dataloader, disable=(self.args.local_rank != 0), desc="Test: "):
            tensors = [mask, hm, radio]
            mask, hm, radio = [torch.from_numpy(tensor).to(self.device) for tensor in tensors]

            pred_mask, pred_hm, pred_tag = self.model(radio)

            pred_mask = pred_mask[:, self.args.video_frames // 2]
            mask = mask[:, self.args.video_frames // 2]
            self.mask_metric.update(pred_mask, mask)

            pred_hm = pred_hm[:, self.args.video_frames // 2].cpu().numpy()
            pred_tag = pred_tag[:, self.args.video_frames // 2].cpu().numpy()
            kp = [frames[self.args.video_frames // 2] for frames in kp]

            pred_kp = []
            for per_hm, per_tag, per_kp, per_edge in zip(pred_hm, pred_tag, kp, edge):
                group, conf = self.keypoint_group.group(per_hm, per_tag)
                self.keypoint_metric.update(group, conf, per_kp, per_edge)
                pred_kp.append(group)

            batch_videos.extend(video)
            batch_gt_mask.append(mask.cpu().numpy())
            batch_pred_mask.append(pred_mask.cpu().numpy())
            batch_gt_kp.extend(kp)
            batch_pred_kp.extend(pred_kp)

        mask_ap, mask_ap_50, mask_ap_60, mask_ap_70, mask_ap_80 = self.mask_metric.compute()
        kp_ap, kp_ap_50, kp_ap_60, kp_ap_70, kp_ap_80, kp_ar = self.keypoint_metric.compute()
        if self.args.local_rank == 0:
            if self.args.wandb_enable:
                wandb.log({
                    "test/mask AP": mask_ap,
                    "test/mask AP@.5": mask_ap_50,
                    "test/mask AP@.6": mask_ap_60,
                    "test/mask AP@.7": mask_ap_70,
                    "test/mask AP@.8": mask_ap_80,
                    "test/keypoint AP": kp_ap,
                    "test/keypoint AP@.5": kp_ap_50,
                    "test/keypoint AP@.6": kp_ap_60,
                    "test/keypoint AP@.7": kp_ap_70,
                    "test/keypoint AP@.8": kp_ap_80,
                    "test/keypoint AR": kp_ar,
                })
            self.logger.info(
                f"Test Result:\n"
                f"---Mask--- AP: {mask_ap * 100:.2f}, AP@.5: {mask_ap_50 * 100:.2f}, AP@.6: {mask_ap_60 * 100:.2f}, AP@.7: {mask_ap_70 * 100:.2f}, AP@.8: {mask_ap_80 * 100:.2f}\n"
                f"---Keypoint--- AP: {kp_ap * 100:.2f}, AP@.5: {kp_ap_50 * 100:.2f}, AP@.6: {kp_ap_60 * 100:.2f}, AP@.7: {kp_ap_70 * 100:.2f}, AP@.8: {kp_ap_80 * 100:.2f}, AR: {kp_ar * 100:.2f}")

        self.mask_metric.reset()
        self.keypoint_metric.reset()

        if self.args.do_test_visual:
            batch_gt_mask = np.concatenate(batch_gt_mask, axis=0)
            batch_pred_mask = np.concatenate(batch_pred_mask, axis=0)

            if self.samples is None:
                num_sample = min(self.args.test_visual_sample, len(batch_videos)) if self.args.test_visual_sample > 0 else len(batch_videos)
                self.samples = random.sample(range(len(batch_videos)), num_sample)

            self.mask_visualizer.test_visual(self.samples, batch_videos, batch_gt_mask, batch_pred_mask)
            self.keypoint_visualizer.test_visual(self.samples, batch_videos, batch_gt_kp, batch_pred_kp)

