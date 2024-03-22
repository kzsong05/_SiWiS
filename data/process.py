from utils.mask import MaskAnnoInferencer, MaskHeatMapGenerator
from utils.keypoint import KeypointInferencer
from utils.generate import DatasetGenerator
from utils.split import DatasetSplit


class Arguments(object):
    def __init__(self):
        self.antennas_num = 4
        self.video_frames = 25
        self.radio_frames = 1024
        self.video_frames_gap = 3
        self.path_list = []

        assert self.video_frames % 2 == 1


if __name__ == "__main__":
    args = Arguments()

    print("Mask Segmentation Inference...")
    mask_inferencer = MaskAnnoInferencer()
    mask_inferencer.inference(args.path_list)
    heatmap_generator = MaskHeatMapGenerator()
    heatmap_generator.generate(args.path_list)

    print("KeyPoint Inference...")
    keypoint_inferencer = KeypointInferencer()
    keypoint_inferencer.inference(args.path_list)

    print("Dataset Generate...")
    dataset_generator = DatasetGenerator(
        antennas_num=args.antennas_num,
        video_frames=args.video_frames,
        radio_frames=args.radio_frames,
        video_frames_gap=args.video_frames_gap,
    )
    dataset_generator.generate(args.path_list)

    print("Dataset Split...")
    dataset_split = DatasetSplit(
        train_ratio=0.9,
        test_ratio=0.1
    )
    dataset_split.process(args.path_list)
