import os
from datetime import datetime


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def check_model_save_path(args):
    if args.model_save_path:
        return args.model_save_path
    elif args.wandb_enable:
        return f"./weights/{args.project_name}-{args.display_name}"
    else:
        return datetime.now().strftime("./weights/%Y-%m-%d-%H-%M-%S")


def check_visual_save_path(args):
    if args.visual_save_path:
        return args.visual_save_path
    elif args.model_save_path:
        return os.path.join("visual", args.model_save_path)
    elif args.wandb_enable:
        return f"./visual/{args.project_name}-{args.display_name}"
    else:
        return datetime.now().strftime("./visual/%Y-%m-%d-%H-%M-%S")
