import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

# ----- Helpers -----

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, unflatten_size):
        super().__init__()
        if isinstance(unflatten_size, tuple):
            self.c = unflatten_size[0]
            self.h = unflatten_size[1]
            self.w = unflatten_size[2]
            self.d = unflatten_size[3]
        elif isinstance(unflatten_size, int):
            self.c = unflatten_size
            self.h = 1
            self.w = 1
            self.d = 1

    def forward(self, x):
        return x.view(x.size(0), self.c, self.h, self.w, self.d)

# ----- 3D Convolutions -----

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weightnorm=True, act=None, drop_prob=0.0):
        super().__init__()
        self.weightnorm = weightnorm

        # Changed to 3D convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.act = nn.ELU(inplace=True) if act is not None else Identity()
        self.drop_prob = drop_prob

        if self.weightnorm:
            self.conv = nn.utils.parametrizations.weight_norm(self.conv, dim=0, name="weight")

    def forward(self, input):
        return F.dropout(self.act(self.conv(input)), p=self.drop_prob, training=True)

class ConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, output_padding=0, dilation=1, groups=1, bias=True, 
                 weightnorm=True, act=None, drop_prob=0.0):
        super().__init__()
        self.weightnorm = weightnorm

        # Changed to 3D transposed convolution
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                      stride=stride, padding=padding,
                                      output_padding=output_padding,
                                      dilation=dilation, groups=groups, bias=bias)

        self.act = nn.ELU(inplace=True) if act is not None else Identity()
        self.drop_prob = drop_prob

        if self.weightnorm:
            self.conv = nn.utils.parametrizations.weight_norm(self.conv, dim=1, name="weight")

    def forward(self, input):
        return F.dropout(self.act(self.conv(input)), p=self.drop_prob, training=True)

# ----- 3D Up/Down Sampling -----

class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.core_nn = nn.Sequential(
            Conv3d(in_channels, out_channels,
                   kernel_size=3, stride=2, padding=1, drop_prob=drop_prob)
        )

    def forward(self, input):
        return self.core_nn(input)
    
class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.core_nn = nn.Sequential(
            ConvTranspose3d(in_channels, out_channels,
                            kernel_size=3, stride=2, padding=1,
                            output_padding=1, drop_prob=drop_prob)
        )

    def forward(self, input):
        return self.core_nn(input)
    
# ----- 3D Channel Attention -----

class CALayer3D(nn.Module):
    def __init__(self, channel, reduction=8, drop_prob=0.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D pooling
        
        self.ca_block = nn.Sequential(
            Conv3d(channel, channel // reduction, 
                   kernel_size=1, act=nn.ELU(), drop_prob=drop_prob),
            Conv3d(channel // reduction, channel,
                   kernel_size=1, act=None, drop_prob=drop_prob),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca_block(y)
        return x * y

# ----- 3D Dense Blocks -----

class DenseNetBlock3D(nn.Module):
    def __init__(self, inplanes, growth_rate, drop_prob=0.0):
        super().__init__()
        self.dense_block = nn.Sequential(
            Conv3d(inplanes, 4 * growth_rate,
                   kernel_size=1, stride=1, padding=0, drop_prob=drop_prob),
            Conv3d(4 * growth_rate, growth_rate,
                   kernel_size=3, stride=1, padding=1, drop_prob=drop_prob, act=None)
        )

    def forward(self, input):
        y = self.dense_block(input)
        y = torch.cat([input, y], dim=1)
        return y
    
class DenseNetLayer3D(nn.Module):
    def __init__(self, inplanes, growth_rate, steps, drop_prob=0.0):
        super().__init__()
        self.activation = nn.ELU(inplace=True)

        net = []
        for step in range(steps):
            net.append(DenseNetBlock3D(inplanes, growth_rate, drop_prob=drop_prob))
            net.append(self.activation)
            inplanes += growth_rate

        net.append(CALayer3D(inplanes, drop_prob=drop_prob))
        self.core_nn = nn.Sequential(*net)

    def forward(self, input):
        return self.core_nn(input)
    
class DenselyNetwork3D(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, steps, blocks, act=None, drop_prob=0.0):
        super().__init__()
        # downscale block
        net = []
        for i in range(blocks):
            net.append(DenseNetLayer3D(in_channels, growth_rate, steps, drop_prob=drop_prob))
            in_channels = in_channels + growth_rate * steps

        # output layer
        net.append(Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=None))

        self.core_nn = nn.Sequential(*net)            

    def forward(self, input):
        return self.core_nn(input)
    
# ----- 3D Encoder/Decoder -----

class DenselyEncoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, steps, scale_factor, drop_prob=0.0):
        super().__init__()
        # downscale block
        net = []
        for i in range(scale_factor):
            net.append(DenseNetLayer3D(in_channels, growth_rate, steps, drop_prob=drop_prob))
            in_channels = in_channels + growth_rate * steps
            net.append(Downsample3D(in_channels, 2*in_channels, drop_prob=drop_prob))
            in_channels *= 2
            growth_rate *= 2

        # output block
        net.append(DenseNetLayer3D(in_channels, growth_rate, steps, drop_prob=drop_prob))
        in_channels = in_channels + growth_rate * steps

        # output layer
        net.append(Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=None))

        self.core_nn = nn.Sequential(*net)

    def forward(self, input):
        return self.core_nn(input)

class DenselyDecoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=16, steps=3, scale_factor=2, drop_prob=0.0):
        super().__init__()
        # upsample block
        net = []
        for i in range(scale_factor):
            net.append(DenseNetLayer3D(in_channels, growth_rate, steps, drop_prob=drop_prob))
            in_channels = in_channels + growth_rate * steps
            net.append(Upsample3D(in_channels, in_channels//2, drop_prob=drop_prob))
            in_channels = in_channels//2
            growth_rate = growth_rate//2

        # output block
        net.append(Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=None))

        self.core_nn = nn.Sequential(*net)

    def forward(self, x):
        return self.core_nn(x)