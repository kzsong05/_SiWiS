import os
import shutil
from tqdm import tqdm
from utils.basic import get_path_list
from mmpose.apis import MMPoseInferencer


class MMDET(object):
    def __init__(self):
        self.mmdet_dict = {
            'YOLOX-l': {
                'data_config': 'utils/resource/yolox_l_8x8_300e_coco.py',
                'model': 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
                'id': [0],
            },
        }

    # Get model configuration according to the input name
    def get(self, name):
        if name not in self.mmdet_dict:
            raise Exception("Detection Model Not Exist!")
        config = self.mmdet_dict[name]['data_config']
        model = self.mmdet_dict[name]['model']
        id = self.mmdet_dict[name]['id']
        return config, model, id


class MMPOSE(object):
    def __init__(self):
        self.mmpose_dict = {
            "HRNET": {
                'data_config': 'utils/resource/td-hm_hrnet-w48_8xb32-210e_coco-384x288.py',
                'model': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth',
            }
        }

    # Get model configuration according to the input name
    def get(self, name):
        if name not in self.mmpose_dict:
            raise Exception("Pose Model Not Exist!")
        config = self.mmpose_dict[name]['data_config']
        model = self.mmpose_dict[name]['model']
        return config, model


class KeypointInferencer(object):
    def __init__(self):
        mmdet = MMDET()
        det_model, det_weights, det_cat_ids = mmdet.get("YOLOX-l")
        mmpose = MMPOSE()
        pose2d, pose2d_weights = mmpose.get("HRNET")

        self.pose_inferencer = MMPoseInferencer(
            pose2d=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=det_cat_ids,
        )

    def inference(self, path_list=None):
        path_list = get_path_list(path_list)

        for path in path_list:
            video_path = os.path.join(path, "video")
            out_path = os.path.join(path, "keypoint")
            if os.path.exists(out_path):
                shutil.rmtree(out_path, ignore_errors=True)
            os.mkdir(out_path)

            inferencer = self.pose_inferencer(video_path, show=False, return_vis=False, pred_out_dir=out_path)

            for result in tqdm(inferencer, total=len(os.listdir(video_path)), desc=f"Extract Keypoint {video_path}: "):
                pass
