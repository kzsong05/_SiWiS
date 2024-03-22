import os
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from configs.config import BasicConfigs
from utils.basic import seeds_init, wandb_init, Logger, model_params_count
from dataloader.dataset import BasicDataset
from dataloader.dataloader import BasicDataloader
from metric.iou import IOUMetric
from metric.oks import OKSMetric
from model.amsen import Amsen
from trainer.trainer import Trainer
from trainer.evaluate import Evaluator


def run(args):
    train_dataset = BasicDataset(
        data_file=args.train_file,
        num_antennas=args.num_antennas,
        video_frames=args.video_frames,
        radio_frames=args.radio_frames,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        num_keypoint=args.num_keypoint,
    )
    train_dataloader = BasicDataloader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        workers_per_gpu=args.workers_per_gpu,
    ).get_train_dataloader()

    test_dataset = BasicDataset(
        data_file=args.test_file,
        num_antennas=args.num_antennas,
        video_frames=args.video_frames,
        radio_frames=args.radio_frames,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        num_keypoint=args.num_keypoint,
    )
    test_dataloader = BasicDataloader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        workers_per_gpu=args.workers_per_gpu,
    ).get_test_dataloader()

    mask_metric = IOUMetric(
        heatmap_size=args.heatmap_size,
        pred_mask_threshold=args.pred_mask_threshold
    )
    keypoint_metric = OKSMetric(
        device=args.device,
        num_keypoint=args.num_keypoint,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
    )

    model = Amsen(
        num_antennas=args.num_antennas,
        video_frames=args.video_frames,
        radio_frames=args.radio_frames,
        num_keypoint=args.num_keypoint,
        norm_num_groups=args.norm_num_groups,
    )
    model.mask_metric = mask_metric
    model.keypoint_metric = keypoint_metric
    model.to(args.device)
    args.trainable_model_params = model_params_count(model)

    training_parameters = []
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    training_parameters.append({"params": ddp_model.parameters(), "lr": args.learning_rate})

    if not len(training_parameters):
        raise Exception("No parameter trainable!")

    optimizer = torch.optim.AdamW(training_parameters)
    optimizer.zero_grad()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scheduler_gamma)

    evaluator = Evaluator(
        args=args,
        model=ddp_model,
        mask_metric=mask_metric,
        keypoint_metric=keypoint_metric,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader
    )

    trainer = Trainer(
        args=args,
        model=ddp_model,
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        evaluator=evaluator
    )
    trainer.train()


def main():
    configs = BasicConfigs()
    args = configs.get_args()
    seeds_init(args.seed)

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    }
    args.rank = int(env_dict["RANK"])
    args.local_rank = int(env_dict["LOCAL_RANK"])
    args.world_size = int(env_dict["WORLD_SIZE"])
    args.device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else "cpu"

    logger = Logger(__name__).get_logger(args.log_file)
    if args.local_rank == 0:
        if args.wandb_enable:
            wandb_init(args)
        logger.info("=" * 80)
        logger.info("Opts".center(80))
        logger.info("-" * 80)
        for key, item in args.items():
            logger.info("{:<35s}: {:<35s}".format(str(key), str(item)).center(80))
        logger.info("=" * 80)

    dist.init_process_group(backend="nccl")
    run(args)
    dist.destroy_process_group()
    if args.wandb_enable:
        wandb.finish()


if __name__ == "__main__":
    main()



