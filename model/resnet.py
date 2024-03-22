from torch import nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
        norm_num_groups: int = 32,
        act_fn: bool = True,
    ):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )

        self.norm = nn.GroupNorm(min(norm_num_groups, out_channels), out_channels)
        self.act_fn = nn.ReLU() if act_fn else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        return hidden_state


class ShortCut(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(ShortCut, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        return hidden_state


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
    ):
        super(ResNet, self).__init__()

        self.conv_layers = nn.Sequential(
            ConvLayer(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=(1, 1), padding=(0, 0), norm_num_groups=norm_num_groups),
            ConvLayer(in_channels=in_channels // 4, out_channels=in_channels // 4, norm_num_groups=norm_num_groups),
            nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels, kernel_size=(1, 1), bias=False),
        )

        should_apply_shortcut = in_channels != out_channels
        self.shortcut = ShortCut(in_channels, out_channels) if should_apply_shortcut else nn.Identity()
        self.norm = nn.GroupNorm(min(norm_num_groups, out_channels), out_channels)
        self.act_fn = nn.ReLU()

    def forward(self, input):
        residual = self.shortcut(input)
        hidden_state = self.conv_layers(input)
        hidden_state += residual
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        return hidden_state


class UpSample(nn.Module):
    def __init__(
        self,
        scale_factor: int = 2,
    ):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, hidden_state):
        hidden_state = nn.functional.interpolate(hidden_state, scale_factor=self.scale_factor, mode="nearest")
        return hidden_state
