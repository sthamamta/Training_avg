import torch
import torch.nn.functional as F
from torch import nn


# __all__ = [
#     "Discriminator", "RRDBNet", "ContentLoss",
#     "discriminator", "rrdbnet_x1", "rrdbnet_x2", "rrdbnet_x4", "rrdbnet_x8", "content_loss",
# ]


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out
class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            channels: int = 16,
            growth_channels: int = 8,
            num_blocks: int = 12,
            upscale_factor: int = 2,
            mode = 'nearest',
    ) -> None:
        super(RRDBNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.growth_channels = growth_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor
        self.mode= mode


        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        

        # Upsampling convolutional layer.
        if upscale_factor == 2:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
        if upscale_factor == 4:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
        if upscale_factor == 8:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling3 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv_first = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        self.conv_second = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        self.conv_third = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        self._initialize_weights()

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        if self.upscale_factor == 2:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode=self.mode))
        if self.upscale_factor == 4:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode=self.mode))
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode=self.mode))
        if self.upscale_factor == 8:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode=self.mode))
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode=self.mode))
            out = self.upsampling3(F.interpolate(out, scale_factor=2, mode=self.mode))

        out = self.conv3(out)

        first = self.conv_first(out)
        second = self.conv_second(out)
        third = self.conv_third(out)

        first = torch.clamp_(first, 0.0, 1.0)
        second = torch.clamp_(second, 0.0, 1.0)
        third = torch.clamp_(third, 0.0, 1.0)

        return first,second,third

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def save(self,model_weights,path,optimizer_weights,epoch):
         torch.save({
                    'in_channels':self.in_channels,
                    'out_channels':self.out_channels,
                    'channels':self.channels,
                    'growth_channels': self.growth_channels,
                    'num_blocks': self.num_blocks,
                    'mode': self.mode,
                    'upscale_factor': self.upscale_factor,
                    'epoch': epoch,
                    'model_state_dict': model_weights,
                    'optimizer_state_dict': optimizer_weights,
                    }, path) 


