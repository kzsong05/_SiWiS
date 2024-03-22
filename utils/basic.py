import os
import torch
import wandb
import random
import logging
import numpy as np
from datetime import datetime
import torch.backends.cudnn as cudnn


def model_params_count(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(total_params / 1e6, 2)


def wandb_init(args):
    project_name = args.project_name
    display_name = args.display_name

    wandb.init(project=project_name, name=display_name, config=args)


def seeds_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class Logger(object):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.log_file = datetime.now().strftime("./logs/%Y-%m-%d-%H-%M-%S.log")

    def get_logger(self, log_file, console_level=logging.INFO, file_level=logging.DEBUG):
        self.log_file = log_file if log_file else self.log_file
        if not os.path.exists(os.path.split(self.log_file)[0]):
            os.makedirs(os.path.split(self.log_file)[0], exist_ok=True)
        self.console_level = console_level
        self.file_level = file_level

        # add a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_format = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                           datefmt="%m/%d/%Y %H:%M:%S")
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # add a file handler
        file_handler = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(self.file_level)
        file_format = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                        datefmt="%m/%d/%Y %H:%M:%S")
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(min(self.console_level, self.file_level))

        return self.logger
