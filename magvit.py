from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from periodic_padding import periodic_padding_3d

def crop_tensor(x):
	x = x.narrow(2,1,x.shape[2]-1).narrow(3,1,x.shape[3]-1).narrow(4,1,x.shape[4]-1).contiguous()
	return x

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

class MagVitAE3D(nn.Module):
    def __init__(
        self,
        n_bands: int = 3,
        hidden_dims: int = 512,
        residual_conv_kernel_size: int = 3,
        n_compressions: int = 2,
        num_consecutive: int = 2,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        init_dim = int(hidden_dims / 2**n_compressions)
        dim = init_dim

        self.conv_in = SameConv3d(n_bands, init_dim, 7)
        self.conv_out = SameConv3d(init_dim, n_bands, 3)

        encoder_layer = ResidualUnit3D(dim, residual_conv_kernel_size)
        decoder_layer = ResidualUnit3D(dim, residual_conv_kernel_size)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        for i in range(n_compressions):
            dim_out = dim * 2
            encoder_layer = SpatialDownsample3D(dim, dim_out)
            decoder_layer = SpatialUpsample3D(dim_out, dim)
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)
            dim = dim_out

            encoder_layer = nn.Sequential(
                *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
            )
            decoder_layer = nn.Sequential(
                *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
            )
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)

        dim_out = dim
        encoder_layer = SameConv3d(dim, dim_out, 7)
        decoder_layer = SameConv3d(dim_out, dim, 3)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        encoder_layer = nn.Sequential(
            *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
        )
        decoder_layer = nn.Sequential(
            *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
        )
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        self.encoder_layers.append(
            nn.Sequential(
                Rearrange("b c ... -> b ... c"),
                nn.LayerNorm(dim),
                Rearrange("b ... c -> b c ..."),
            )
        )

    def encode(self, x: torch.Tensor):
        x = self.conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor):
        i = 0
        for layer in self.decoder_layers:
            if i%2 == 0:
                x = periodic_padding_3d(x,(0,1,0,1,0,1))
                x = layer(x)
                x = crop_tensor(x)
            else:
                x = layer(x)
            i = i+1
        x = self.conv_out(x)
        return x


    def forward(self, x: torch.Tensor):
        # Encoder
        x = self.encode(x)
        # Decoder
        x = self.decode(x)
        return x

class MagVitVAE3D(nn.Module):
    def __init__(
        self,
        n_bands: int = 3,
        hidden_dims: int = 512,
        residual_conv_kernel_size: int = 3,
        n_compressions: int = 2,
        num_consecutive: int = 2,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        init_dim = int(hidden_dims / 2**n_compressions)
        dim = init_dim

        self.conv_in = SameConv3d(n_bands, init_dim, 7)
        self.conv_out = SameConv3d(init_dim, n_bands, 3)

        # Initial residual block
        encoder_layer = ResidualUnit3D(dim, residual_conv_kernel_size)
        decoder_layer = ResidualUnit3D(dim, residual_conv_kernel_size)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        # Compression blocks
        for i in range(n_compressions):
            dim_out = dim * 2
            encoder_layer = SpatialDownsample3D(dim, dim_out)
            decoder_layer = SpatialUpsample3D(dim_out, dim)
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)
            dim = dim_out

            encoder_layer = nn.Sequential(
                *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
            )
            decoder_layer = nn.Sequential(
                *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
            )
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)

        # Final non-compress blocks
        dim_out = dim
        encoder_layer = SameConv3d(dim, dim_out, 7)
        decoder_layer = SameConv3d(dim_out, dim, 3)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        encoder_layer = nn.Sequential(
            *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
        )
        decoder_layer = nn.Sequential(
            *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
        )
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        # Final normalization layer
        self.encoder_layers.append(
            nn.Sequential(
                Rearrange("b c ... -> b ... c"),
                nn.LayerNorm(dim),
                Rearrange("b ... c -> b c ..."),
            )
        )

        # Layers for latent distribution parameters (VAE-specific)
        self.to_mu = nn.Conv3d(dim, dim, kernel_size=1)
        self.to_logvar = nn.Conv3d(dim, dim, kernel_size=1)

    def encode(self, x: torch.Tensor):
        x = self.conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor):
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.conv_out(x)
        return x

    def forward(self, x: torch.Tensor):
        # Encoder
        encoded = self.encode(x)
        # Obtain latent distribution parameters
        mu = self.to_mu(encoded)
        logvar = self.to_logvar(encoded)
        std = torch.exp(0.5 * logvar)
        # Reparameterization trick
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Decoder
        decoded = self.decode(z)
        return decoded, mu, logvar

class MagVitVAE3D(nn.Module):
    def __init__(
        self,
        n_bands: int = 3,
        hidden_dims: int = 512,
        residual_conv_kernel_size: int = 3,
        n_compressions: int = 2,
        num_consecutive: int = 2,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        init_dim = int(hidden_dims / 2**n_compressions)
        dim = init_dim

        self.conv_in = SameConv3d(n_bands, init_dim, 7)
        self.conv_out = SameConv3d(init_dim, n_bands, 3)

        # Initial residual block
        encoder_layer = ResidualUnit3D(dim, residual_conv_kernel_size)
        decoder_layer = ResidualUnit3D(dim, residual_conv_kernel_size)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        # Compression blocks
        for i in range(n_compressions):
            dim_out = dim * 2
            encoder_layer = SpatialDownsample3D(dim, dim_out)
            decoder_layer = SpatialUpsample3D(dim_out, dim)
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)
            dim = dim_out

            encoder_layer = nn.Sequential(
                *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
            )
            decoder_layer = nn.Sequential(
                *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
            )
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)

        # Final non-compress blocks
        dim_out = dim
        encoder_layer = SameConv3d(dim, dim_out, 7)
        decoder_layer = SameConv3d(dim_out, dim, 3)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        encoder_layer = nn.Sequential(
            *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
        )
        decoder_layer = nn.Sequential(
            *[ResidualUnit3D(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
        )
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        # Final normalization layer
        self.encoder_layers.append(
            nn.Sequential(
                Rearrange("b c ... -> b ... c"),
                nn.LayerNorm(dim),
                Rearrange("b ... c -> b c ..."),
            )
        )

        # Layers for latent distribution parameters (VAE-specific)
        self.to_mu = nn.Conv3d(dim, dim, kernel_size=1)
        self.to_logvar = nn.Conv3d(dim, dim, kernel_size=1)

    def encode(self, x: torch.Tensor):
        x = self.conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor):
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.conv_out(x)
        return x

    def forward(self, x: torch.Tensor):
        # Encoder
        encoded = self.encode(x)
        # Obtain latent distribution parameters
        mu = self.to_mu(encoded)
        logvar = self.to_logvar(encoded)
        std = torch.exp(0.5 * logvar)
        # Reparameterization trick
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Decoder
        decoded = self.decode(z)
        return decoded, mu, logvar
     
class SpatialUpsample3D(nn.Module):
    def __init__(self, dim: int, dim_out: int = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * 8, 1)
        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange("b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)", p1=2, p2=2, p3=2),
        )
        self.init_conv_(conv)
    
    def init_conv_(self, conv: nn.Module):
        o, i, d, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 8, i, d, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 8) ...")
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor):
        return self.net(x)

class SpatialDownsample3D(nn.Module):
    def __init__(self, dim: int, dim_out: int = None, kernel_size: int = 3):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.conv = nn.Conv3d(dim, dim_out, kernel_size, stride=2, padding=kernel_size // 2)
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)

class Residual3D(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
    
    def forward(self, x: torch.Tensor, **kwargs):
        return self.fn(x, **kwargs) + x

def ResidualUnit3D(dim: int, kernel_size: Union[int, Tuple[int, int, int]]):
    net = torch.nn.Sequential(
        SameConv3d(dim, dim, kernel_size),
        nn.ELU(),
        nn.Conv3d(dim, dim, 1),
        nn.ELU(),
    )
    return Residual3D(net)

def SameConv3d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 3)
    padding = [k // 2 for k in kernel_size]
    return nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, padding=padding)

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

def default(v, d):
    return v if exists(v) else d

def exists(v):
    return v is not None
