import os
import yaml
from easydict import EasyDict


class BasicConfigs(object):
    def __init__(self):
        args = dict()
        self.args = EasyDict(args)
        self.args.config_file_path = "./configs/config.yaml"
        self._load_config_file()

    def _load_config_file(self):
        assert os.path.exists(self.args.config_file_path)
        with open(self.args.config_file_path, "r", encoding="utf-8") as cfg_file:
            configs = yaml.safe_load(cfg_file)
            for key, value in configs.items():
                self.args[key] = value

    def _check_arguments(self):
        assert self.args.video_frames % 2 == 1

    def get_args(self):
        self._check_arguments()
        return self.args
