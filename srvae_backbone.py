import torch
import torch.nn as nn
import torch.nn.functional as F
from srvae_layer import *
class q_u(nn.Module):
    """ 3D Encoder q(u|y) """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_in = input_shape[0]
        nc_out = 2 * output_shape[0]

        self.core_nn = nn.Sequential(
            DenselyEncoder3D(  # Changed to 3D version
                in_channels=nc_in,
                out_channels=nc_out,
                growth_rate=64,
                steps=3,
                scale_factor=2)
        )

    def forward(self, input):
        mu, logvar = self.core_nn(input).chunk(2, 1)
        #return mu, F.hardtanh(logvar, min_val=-7, max_val=7.)
        return mu, logvar

class p_y(nn.Module):
    """ 3D Decoder p(y|u) """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_in = input_shape[0]
        nc_out = output_shape[0]

        self.core_nn = nn.Sequential(
            DenselyDecoder3D(  # Changed to 3D version
                in_channels=nc_in,
                out_channels=nc_out,
                growth_rate=128,
                steps=4,
                scale_factor=2)
        )

    def forward(self, input):
        logits = self.core_nn(input)
        return logits

class q_z(nn.Module):
    """ 3D Encoder q(z|x) """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_in = input_shape[0]
        nc_out = 2 * output_shape[0]

        self.core_nn = nn.Sequential(
            DenselyEncoder3D(  # Changed to 3D version
                in_channels=nc_in,
                out_channels=nc_out,
                growth_rate=16,
                steps=4,
                scale_factor=2)
        )

    def forward(self, input):
        mu, logvar = self.core_nn(input).chunk(2, 1)
        #return mu, F.hardtanh(logvar, min_val=-7, max_val=7.)
        return mu, logvar

class p_z(nn.Module):
    """ 3D Encoder p(z| y, u) """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_y_in, nc_u_in = input_shape[0][0], input_shape[1][0]
        nc_out = 2 * output_shape[0]

        self.y_nn = nn.Sequential(
            DenselyEncoder3D(  # Changed to 3D
                in_channels=nc_y_in,
                out_channels=nc_out//2,
                growth_rate=32,
                steps=5,
                scale_factor=2),
            nn.ELU(inplace=True)
        )

        self.u_nn = nn.Sequential(
            DenselyNetwork3D(  # Changed to 3D
                in_channels=nc_u_in, 
                out_channels=nc_out//2,
                growth_rate=64,
                steps=3,
                blocks=3,
                act=True)
        )

        self.core_nn = nn.Sequential(
            DenselyNetwork3D(  # Changed to 3D
                in_channels=nc_out,
                out_channels=nc_out,
                growth_rate=64,
                steps=3,
                blocks=3,
                act=None)
        )

    def forward(self, input):
        y, u = input[0], input[1]
        y_out = self.y_nn(y)    
        u_out = self.u_nn(u)
        joint = torch.cat((y_out, u_out), 1)
        mu, logvar = self.core_nn(joint).chunk(2, 1)
        #return mu, F.hardtanh(logvar, min_val=-7, max_val=7.)
        return mu, logvar

class p_x(nn.Module):
    """ 3D Decoder p(x| y, z) """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_y_in, nc_z_in = input_shape[0][0], input_shape[1][0]
        nc_out = output_shape[0]

        self.z_nn = nn.Sequential(
            DenselyDecoder3D(  # Changed to 3D
                in_channels=nc_z_in,
                out_channels=nc_out,
                growth_rate=64,
                steps=8,
                scale_factor=2)
        )

        self.core_nn = nn.Sequential(
            DenselyNetwork3D(  # Changed to 3D
                in_channels=nc_out + 3,
                out_channels=nc_out,
                growth_rate=64,
                steps=5,
                blocks=3,
                act=None)
        )

    def forward(self, input):
        y, z = input[0], input[1]
        # Changed to 3D interpolation
        y_out = y
        z_out = self.z_nn(z)
        joint = torch.cat((y_out, z_out), 1)
        logits = self.core_nn(joint)
        return logits