import os
import torch
from torch import nn
from model.embedding import RadioEmbedding
from model.module import MaskDecoderModule, KeypointDecoderModule


class Amsen(nn.Module):
    def __init__(
        self,
        num_antennas: int,
        video_frames: int,
        radio_frames: int,
        num_keypoint: int,
        norm_num_groups: int,
    ):
        super(Amsen, self).__init__()
        self.video_frames = video_frames
        self.num_keypoint = num_keypoint

        self.embedding = RadioEmbedding(
            num_antennas=num_antennas,
            video_frames=video_frames,
            radio_frames=radio_frames,
        )

        self.mask_decoder = MaskDecoderModule(
            video_frames=video_frames,
            norm_num_groups=norm_num_groups
        )

        self.keypoint_decoder = KeypointDecoderModule(
            video_frames=video_frames,
            num_keypoint=num_keypoint,
            norm_num_groups=norm_num_groups
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def save_pretrained(self, save_path):
        state_file = os.path.join(save_path, "model.pth")
        state = {
            "embedding": self.embedding.state_dict(),
            "mask_decoder": self.mask_decoder.state_dict(),
            "keypoint_decoder": self.keypoint_decoder.state_dict(),
        }
        torch.save(state, state_file)

    def from_pretrained(self, load_path, map_location):
        state_file = os.path.join(load_path, "model.pth")
        if not os.path.exists(state_file):
            print(f"Load Pretrained Model Error: {state_file} not exist")
            return
        state = torch.load(state_file, map_location=map_location)
        self.embedding.load_state_dict(state["embedding"], strict=True)
        self.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        self.keypoint_decoder.load_state_dict(state["keypoint_decoder"], strict=True)

    def forward(self, input):
        hidden_state = self.embedding(input)

        pred_mask = self.mask_decoder(hidden_state)
        pred_mask = pred_mask.squeeze(2)

        pred_keypoint = self.keypoint_decoder(hidden_state)
        pred_hm = pred_keypoint[:, :, :self.num_keypoint, :, :]
        pred_tag = pred_keypoint[:, :, self.num_keypoint:, :, :]
        return pred_mask, pred_hm, pred_tag
