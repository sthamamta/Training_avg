import torch
import torch.nn as nn
import torch.nn.functional as F


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x):
        out = self.upsample_block(x)

        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
        for i in range(num_layers - 1):
            self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size=3))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class SRDenseNet(nn.Module):
    def __init__(self, num_channels=1, growth_rate=2, num_blocks=2, num_layers=2,upscale_factor=2, mode ='bicubic'):
        super(SRDenseNet, self).__init__()

        self.num_channels = num_channels
        self.growth_rate = growth_rate
        self.num_blocks =  num_blocks
        self.num_layers = num_layers
        self.upscale_factor = upscale_factor
        self.model = mode

        # low level features
        self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)
        self.upscale_factor = upscale_factor
        self.mode = mode


        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, growth_rate * num_layers * num_blocks, kernel_size=1),
            nn.LeakyReLU(0.2, True)
        )

        # reconstruction layer
        self.reconstruction1 = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers * num_blocks, num_channels, kernel_size=3, padding=3 // 2),
            nn.Tanh()
        )

        self.reconstruction2 = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers * num_blocks, num_channels, kernel_size=3, padding=3 // 2),
            nn.Tanh()
        )

        self.reconstruction3 = nn.Sequential(
                    nn.Conv2d(growth_rate * num_layers * num_blocks, num_channels, kernel_size=3, padding=3 // 2),
                    nn.Tanh()
                )


        self._initialize_weights()

    def _initialize_weights(self) -> None:
      print('initializing weights of a model')
      for module in self.modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
              nn.init.kaiming_normal_(module.weight)
              # module.weight.data *= 1.2
              if module.bias is not None:
                  nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode= self.mode) # add this layer if output needs to have doble the size of the input
        x1 = self.conv(x)
        x2 = self.dense_blocks(x1)
        x3 = self.bottleneck(x2)
        first = self.reconstruction1(x3)
        second = self.reconstruction2(x3)
        third = self.reconstruction3(x3)
        return first,second,third

    def save(self,model_weights,path,optimizer_weights,epoch):
         torch.save({
                    'num_channels':self.num_channels,
                    'growth_rate': self.growth_rate,
                    'num_blocks': self.num_blocks,
                    'num_layers':self.num_layers,
                    'mode': self.mode,
                    'upscale_factor': self.upscale_factor,
                    'epoch': epoch,
                    'model_state_dict': model_weights,
                    'optimizer_state_dict': optimizer_weights,
                    }, path) 



