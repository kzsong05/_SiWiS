from torch import nn
from einops import rearrange
from model.transformer import LinearLayer, SelfAttnBlock


class RadioEmbedding(nn.Module):
    def __init__(
        self,
        num_antennas: int,
        video_frames: int,
        radio_frames: int,
    ):
        super(RadioEmbedding, self).__init__()

        self.num_antennas = num_antennas
        self.video_frames = video_frames
        self.radio_frames = radio_frames

        self.ant_linear = nn.Sequential(
            LinearLayer(8, 64),
            LinearLayer(64, 32),
        )

        self.radio_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
        )

        self.video_attn = nn.Sequential(
            SelfAttnBlock(512, 512, self.video_frames, 128),
            SelfAttnBlock(512, 512, self.video_frames, 128)
        )

    def forward(self, input):
        hidden_state = input.flatten(-2, -1)
        hidden_state = self.ant_linear(hidden_state)
        hidden_state = rearrange(hidden_state, "N V R A -> (N V) A R")
        hidden_state = self.radio_conv(hidden_state)
        hidden_state = rearrange(hidden_state, "(N V) A R -> N V (A R)", V=self.video_frames)
        hidden_state = self.video_attn(hidden_state)
        return hidden_state
