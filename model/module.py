import torch.nn as nn
from einops import rearrange
from model.resnet import ResNet, UpSample
from model.transformer import PixelAttnBlock, SelfAttnBlock


class MaskDecoderModule(nn.Module):
    def __init__(
        self,
        video_frames: int,
        norm_num_groups: int = 32,
    ):
        super(MaskDecoderModule, self).__init__()
        self.video_frames = video_frames

        self.pixel_attention = PixelAttnBlock(
            in_dims=512,
            hidden_embed_dim=256,
            max_pixel_length=48,
            max_frame_length=self.video_frames,
        )
        self.pixel_attn = SelfAttnBlock(256, 256, 48, 128)

        self.decoder_channels = [256, 256, 128, 64, 32]
        self.decoder_size = [8, 16, 32, 64, 64]

        self.decoder_blocks = nn.ModuleList([])
        in_out_channels = zip(self.decoder_channels, self.decoder_channels[1:])
        in_out_size = zip(self.decoder_size, self.decoder_size[1:])
        for (in_channels, out_channels), (in_size, out_size) in zip(in_out_channels, in_out_size):
            self.decoder_blocks.extend([
                ResNet(in_channels, out_channels, norm_num_groups),
            ])

            if in_size != out_size:
                self.decoder_blocks.append(
                    UpSample(out_size // in_size)
                )

        self.head = nn.Sequential(
            ResNet(32, 32, 8),
            ResNet(32, 16, 4),
            nn.Conv2d(16, 1, 1, bias=False)
        )

    def forward(self, hidden_state):
        hidden_state = self.pixel_attention(hidden_state)
        hidden_state = rearrange(hidden_state, "N V l d -> (N V) l d")
        hidden_state = self.pixel_attn(hidden_state)
        hidden_state = rearrange(hidden_state, "N (h w) C -> N C h w", h=6)

        for block in self.decoder_blocks:
            hidden_state = block(hidden_state)

        hidden_state = self.head(hidden_state)
        hidden_state = rearrange(hidden_state, "(N V) C h w -> N V C h w", V=self.video_frames)
        return hidden_state


class KeypointDecoderModule(nn.Module):
    def __init__(
        self,
        video_frames: int,
        num_keypoint: int,
        norm_num_groups: int = 32,
    ):
        super(KeypointDecoderModule, self).__init__()
        self.video_frames = video_frames
        self.num_keypoint = num_keypoint

        self.pixel_attention = PixelAttnBlock(
            in_dims=512,
            hidden_embed_dim=256,
            max_pixel_length=48,
            max_frame_length=self.video_frames,
        )
        self.pixel_attn = SelfAttnBlock(256, 256, 48, 128)

        self.decoder_channels = [256, 256, 128, 64, 32]
        self.decoder_size = [8, 16, 32, 64, 64]

        self.decoder_blocks = nn.ModuleList([])
        in_out_channels = zip(self.decoder_channels, self.decoder_channels[1:])
        in_out_size = zip(self.decoder_size, self.decoder_size[1:])
        for (in_channels, out_channels), (in_size, out_size) in zip(in_out_channels, in_out_size):
            self.decoder_blocks.extend([
                ResNet(in_channels, out_channels, norm_num_groups),
            ])

            if in_size != out_size:
                self.decoder_blocks.append(
                    UpSample(out_size // in_size)
                )

        self.head = nn.Sequential(
            ResNet(32, 32, 8),
            ResNet(32, 104, 13),
            nn.Conv2d(104, 26, 1, bias=False)
        )

    def forward(self, hidden_state):
        hidden_state = self.pixel_attention(hidden_state)
        hidden_state = rearrange(hidden_state, "N V l d -> (N V) l d")
        hidden_state = self.pixel_attn(hidden_state)
        hidden_state = rearrange(hidden_state, "N (h w) C -> N C h w", h=6)

        for block in self.decoder_blocks:
            hidden_state = block(hidden_state)

        hidden_state = self.head(hidden_state)
        hidden_state = rearrange(hidden_state, "(N V) C h w -> N V C h w", V=self.video_frames)
        return hidden_state

