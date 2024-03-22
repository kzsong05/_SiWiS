import os
import torch
import wandb
from tqdm import tqdm
from utils.basic import Logger
from utils.check import check_path, check_model_save_path
from loss.mask import MaskLoss
from loss.keypoint import KeypointLoss


class Trainer(object):
    def __init__(
        self,
        args,
        model,
        evaluator,
        optimizer,
        lr_scheduler,
        train_dataset,
        train_dataloader
    ):
        self.args = args
        self.device = self.args.device
        self.model_save_path = check_model_save_path(args)
        self.model_load_path = self.args.model_load_path
        self.logger = Logger(__name__).get_logger(self.args.log_file)

        self.model = model

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator

        self.mask_loss = MaskLoss(
            bce_loss_weight=self.args.bce_loss_weight,
            dice_loss_weight=self.args.dice_loss_weight,
            heatmap_size=self.args.heatmap_size,
            video_frames=self.args.video_frames,
        )
        self.keypoint_loss = KeypointLoss(
            device=self.args.device,
            mse_loss_weight=self.args.mse_loss_weight,
            within_loss_weight=self.args.within_loss_weight,
            across_loss_weight=self.args.across_loss_weight,
            image_size=self.args.image_size,
            heatmap_size=self.args.heatmap_size,
        )

        self.reload_epoch = 0
        if self.model_load_path is not None:
            self._load_checkpoint(self.model_load_path)

    def _save_checkpoint(self, epoch):
        output_path = os.path.join(self.model_save_path, f"checkpoint-{epoch}")
        check_path(output_path)
        self.logger.info(f"***** Saving model checkpoint {epoch} to {output_path}")
        self.model.module.save_pretrained(output_path)

        optimizer_state = {"optimizer": self.optimizer.state_dict(),
                           "lr_scheduler": self.lr_scheduler.state_dict(),
                           "epoch": epoch}
        torch.save(optimizer_state, os.path.join(output_path, "optimizer.pth"))

    def _load_checkpoint(self, path):
        if self.args.local_rank == 0:
            self.logger.info(f"***** Load model from path {path}")

        map_location = {"cuda:0": f"cuda:{self.args.local_rank}"}
        self.model.module.from_pretrained(path, map_location)

        if os.path.exists(os.path.join(path, "optimizer.pth")):
            optimizer_state = torch.load(os.path.join(path, "optimizer.pth"), map_location=map_location)
            self.optimizer.load_state_dict(optimizer_state["optimizer"])
            self.lr_scheduler.load_state_dict(optimizer_state["lr_scheduler"])
            self.reload_epoch = optimizer_state["epoch"] + 1

    def train(self):
        if self.args.do_train:
            if self.args.local_rank == 0:
                self.logger.info("***** Run training *****")
                self.logger.info(f"  Num examples = {len(self.train_dataset)}")
                self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
                self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
                self.logger.info(f"  Total train batch size = {self.args.world_size * self.args.train_batch_size}")
                self.logger.info(f"  Trainable Model Params = {self.args.trainable_model_params} M")

            for epoch in range(self.reload_epoch, self.args.num_train_epochs):
                self._train_epoch(epoch)

                if (epoch + 1) % self.args.save_epochs == 0:
                    torch.cuda.empty_cache()
                    self.evaluator.eval(epoch)
                    torch.cuda.empty_cache()

                    if self.args.local_rank == 0:
                        self._save_checkpoint(epoch)

        if self.args.do_test:
            torch.cuda.empty_cache()
            self.evaluator.test()

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_dataloader.sampler.set_epoch(epoch)

        for step, (video, mask, hm, kp, edge, radio) in tqdm(enumerate(self.train_dataloader), disable=(self.args.local_rank != 0), total=len(self.train_dataloader), desc=f"Training epoch {epoch}"):
            self.optimizer.zero_grad()

            tensors = [mask, hm, radio]
            mask, hm, radio = [torch.from_numpy(tensor).to(self.device) for tensor in tensors]
            pred_mask, pred_hm, pred_tag = self.model(radio)

            mask_loss = self.mask_loss(pred_mask, mask)
            mse_loss, within_loss, across_loss = self.keypoint_loss(pred_hm, pred_tag, hm, kp)
            keypoint_loss = mse_loss + within_loss + across_loss
            total_loss = mask_loss + keypoint_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            if self.args.local_rank == 0 and self.args.wandb_enable:
                wandb.log({"train/mask_loss": mask_loss.item(),
                           "train/mse_loss": mse_loss.item(),
                           "train/within_loss": within_loss.item(),
                           "train/across_loss": across_loss.item(),
                           "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                           "train/epoch": epoch},
                          step=epoch * len(self.train_dataloader) + step)

        self.lr_scheduler.step()
