import os
import json
import shutil
import traceback
import numpy as np
from tqdm import tqdm
from mmdet.apis import DetInferencer
from pycocotools import mask as cocomask
from utils.basic import get_path_list, get_video_list
from concurrent.futures import ThreadPoolExecutor, as_completed


class MMDET(object):
    def __init__(self):
        self.mmdet_dict = {
            'mask-rcnn': "mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco",
        }

    def get(self, name):
        return self.mmdet_dict[name]


class MaskAnnoInferencer(object):
    def __init__(self):
        mmdet = MMDET()
        det_model = mmdet.get("mask-rcnn")
        self.inferencer = DetInferencer(model=det_model)

    def inference(self, path_list=None):
        path_list = get_path_list(path_list)

        for path in path_list:
            print(f"Start Mask Inference {path}...")
            video_path = os.path.join(path, "video")
            out_path = os.path.join(path, "mask")
            if os.path.exists(out_path):
                shutil.rmtree(out_path, ignore_errors=True)
            os.mkdir(out_path)

            self.inferencer(
                video_path,
                batch_size=32,
                draw_pred=False,
                no_save_vis=True,
                no_save_pred=False,
                out_dir=out_path,
            )
            print(f"Finish Mask Inference {out_path}!")


class MaskHeatMapGenerator(object):
    def __init__(
        self,
    ):
        self.num_threads = 32

    def _multi_thread_generator(self, video_list, path):
        if os.path.exists(os.path.join(path, "mask", "heatmap")):
            shutil.rmtree(os.path.join(path, "mask", "heatmap"), ignore_errors=True)
        os.makedirs(os.path.join(path, "mask", "heatmap"), exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            thread_list = []

            for idx, frame_file in tqdm(video_list, desc=f"Mask Heatmap Generate {path} Submit: "):
                thread = executor.submit(_write_mask, idx, path)
                thread_list.append(thread)

            for process in tqdm(as_completed(thread_list), total=len(thread_list), desc=f"Mask Heatmap Generate {path} Complete: "):
                pass

    def generate(self, path_list=None):
        path_list = get_path_list(path_list)
        for path in path_list:
            video_list = get_video_list(path)
            self._multi_thread_generator(video_list, path)


def _write_mask(idx, path):
    try:
        mask_file = os.path.join(path, "mask", "preds", f"{idx}.json")
        with open(mask_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        masks = cocomask.decode(data["masks"]).astype(np.float32)
        mask_filter = []
        for label_idx, label in enumerate(data["labels"]):
            if label == 0:
                mask_filter.append(masks[:, :, label_idx])

        if not mask_filter:
            mask_size = data["masks"][0]["size"]
            mask_filter.append(np.zeros((mask_size[0], mask_size[1])))

        target = np.stack(mask_filter, axis=0)
        target = np.max(target, axis=0)

        target_file = os.path.join(path, "mask", "heatmap", f"{idx}.npy")
        np.save(target_file, target > 0.5, allow_pickle=True)
        return target_file
    except:
        print(traceback.format_exc())
