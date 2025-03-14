import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from srvae_layer import *
from enum import IntEnum

class Prior(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, **kwargs):
        raise NotImplementedError

    def log_p(self, input, **kwargs):
        return self.forward(z)

    def forward(self, input, **kwargs):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


import numpy as np
from torch.autograd import Variable


# Modified vertion of: https://github.com/divymurli/VAEs

class MixtureOfGaussians(Prior):
    def __init__(self, z_shape, num_mixtures=1000):
        super().__init__()
        self.z_shape = z_shape
        self.z_dim = np.prod(z_shape)
        self.k = num_mixtures

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                / np.sqrt(self.k * self.z_dim))

        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(self.k) / self.k,
                                    requires_grad=False)

    def sample_gaussian(self, m, v):
        """ Element-wise application reparameterization trick to sample from Gaussian
        """
        sample = torch.randn(m.shape)
        z = m + (v**0.5)*sample
        return z

    def log_sum_exp(self, x, dim=0):
        """ Compute the log(sum(exp(x), dim)) in a numerically stable manner
        """
        max_x = torch.max(x, dim)[0]
        new_x = x - max_x.unsqueeze(dim).expand_as(x)
        return max_x + (new_x.exp().sum(dim)).log()

    def log_mean_exp(self, x, dim):
        """ Compute the log(mean(exp(x), dim)) in a numerically stable manner
        """
        return self.log_sum_exp(x, dim) - np.log(x.size(dim))

    def log_normal(self, x, m, v):
        """ Computes the elem-wise log probability of a Gaussian and then sum over the
            last dim. Basically we're assuming all dims are batch dims except for the
            last dim.
        """
        const   = -0.5 * x.size(-1) * torch.log(2*torch.tensor(np.pi))
        log_det = -0.5 * torch.sum(torch.log(v), dim = -1)
        log_exp = -0.5 * torch.sum((x - m)**2/v, dim = -1)

        log_prob = const + log_det + log_exp
        return log_prob

    def log_normal_mixture(self, z, m, v):
        """ Computes log probability of a uniformly-weighted Gaussian mixture.
        """
        z = z.reshape(z.shape[0], 1, -1)
        log_probs = self.log_normal(z, m, v)
        log_prob = self.log_mean_exp(log_probs, 1)
        return log_prob

    def gaussian_parameters(self, h, dim=-1):
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def sample(self, n_samples=1, **kwargs):
        idx = torch.distributions.categorical.Categorical(self.pi).sample((n_samples,))
        m, v = self.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        m, v = m[idx], v[idx]
        z_samples = self.sample_gaussian(m, v)
        return z_samples.reshape(z_samples.shape[0], *self.z_shape)

    def log_p(self, z, **kwargs):
        return self.forward(z)

    def forward(self, z, dim=None, **kwargs):
        """
        Computes the mixture of Gaussian prior
        """
        m, v  = self.gaussian_parameters(self.z_pre, dim=1)
        log_p_z = self.log_normal_mixture(z, m, v)
        return log_p_z

    def __str__(self):
      return "MixtureOfGaussians"


class StandardNormal:
    def __init__(self, z_shape):
        self.z_shape = z_shape

    def sample(self, n_samples=1, **kwargs):
        return torch.randn((n_samples, *self.z_shape))

    def log_p(self, z, **kwargs):
        return self.forward(z)

    def forward(self, z, **kwargs):
        """ Outputs the log p(z).
        """
        log_probs = z.pow(2) + math.log(math.pi * 2.)
        log_probs = -0.5 * log_probs.reshape(z.size(0), -1).sum(dim=1)
        return log_probs

    def __call__(self, z, **kwargs):
        return self.forward(z, **kwargs)

    def __str__(self):
      return "StandardNormal"

class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1

def squeeze_2x2x2(x, reverse=False, alt_order=False):
    """
    For each 3D spatial position, a sub-volume of shape `1x1x1x(N^3 * C)`
    is reshaped into a sub-volume of shape `NxNxNxC`, where N = block_size (2 here).

    This function rearranges a 5D tensor (B, C, D, H, W) such that:
      - When reverse=False (squeeze), the spatial dims (D, H, W) are reduced by a factor of 2
        while the number of channels is multiplied by 8.
      - When reverse=True (unsqueeze), the spatial dims are increased by a factor of 2
        while the channels are divided by 8.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        reverse (bool): Whether to perform the reverse operation (unsqueeze).
        alt_order (bool): Whether to use an alternate ordering via convolution.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    block_size = 2
    if alt_order:
        # Convolution-based approach.
        if x.dim() != 5:
            raise ValueError("Input tensor must be 5D (B, C, D, H, W)")
        b, c, d, h, w = x.size()
        if reverse:
            if c % 8 != 0:
                raise ValueError('For reverse, number of channels must be divisible by 8, got {}.'.format(c))
            new_c = c // 8
        else:
            if d % 2 != 0 or h % 2 != 0 or w % 2 != 0:
                raise ValueError('Depth, height and width must be divisible by 2, got {}x{}x{}'.format(d, h, w))
            new_c = c

        # Build a squeeze matrix for a 2x2x2 block.
        # This is a (8, 1, 2, 2, 2) tensor where each slice has one nonzero element
        # selecting a unique voxel from the 2x2x2 block.
        squeeze_matrix = torch.zeros((8, 1, block_size, block_size, block_size),
                                     dtype=x.dtype, device=x.device)
        squeeze_matrix[0, 0, 0, 0, 0] = 1.
        squeeze_matrix[1, 0, 0, 0, 1] = 1.
        squeeze_matrix[2, 0, 0, 1, 0] = 1.
        squeeze_matrix[3, 0, 0, 1, 1] = 1.
        squeeze_matrix[4, 0, 1, 0, 0] = 1.
        squeeze_matrix[5, 0, 1, 0, 1] = 1.
        squeeze_matrix[6, 0, 1, 1, 0] = 1.
        squeeze_matrix[7, 0, 1, 1, 1] = 1.

        # Build the permutation weights.
        # For each channel, we replicate the 8-slice matrix.
        perm_weight = torch.zeros((8 * new_c, new_c, block_size, block_size, block_size),
                                  dtype=x.dtype, device=x.device)
        for c_idx in range(new_c):
            slice_0 = slice(c_idx * 8, (c_idx + 1) * 8)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :, :] = squeeze_matrix
        # Shuffle the channel order to ensure a consistent reordering.
        shuffle_channels = torch.cat([torch.arange(c_idx * 8, c_idx * 8 + 8) for c_idx in range(new_c)], dim=0)
        perm_weight = perm_weight[shuffle_channels, :, :, :, :]

        # Apply convolution or its transpose.
        if reverse:
            x = F.conv_transpose3d(x, perm_weight, stride=2)
        else:
            x = F.conv3d(x, perm_weight, stride=2)
    else:
        # View/permute method.
        if x.dim() != 5:
            raise ValueError("Input tensor must be 5D (B, C, D, H, W)")
        b, c, d, h, w = x.size()
        # Bring channel to the last dimension.
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        if reverse:
            if c % 8 != 0:
                raise ValueError('For reverse, number of channels {} is not divisible by 8'.format(c))
            new_c = c // 8
            # Reshape to split channels into a factor of 8 and the remaining channels.
            x = x.reshape(b, d, h, w, new_c, block_size, block_size, block_size)
            # Permute to interlace the factors with the spatial dimensions.
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
            # Merge factors with spatial dims: resulting in (B, D*2, H*2, W*2, new_c)
            x = x.contiguous().reshape(b, d * 2, h * 2, w * 2, new_c)
        else:
            if d % 2 != 0 or h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims D x H x W, got {}x{}x{}'.format(d, h, w))
            # Split each spatial dimension by 2.
            x = x.reshape(b, d // 2, block_size, h // 2, block_size, w // 2, block_size, c)
            # Permute to bring the block factors into the channel dimension.
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
            # Merge the factors: new shape (B, D//2, H//2, W//2, C * 8)
            x = x.contiguous().reshape(b, d // 2, h // 2, w // 2, c * 8)
        # Bring channels to the second dimension.
        x = x.permute(0, 4, 1, 2, 3)  # (B, new_channels, D, H, W)
    return x

def checkerboard_mask_3d(depth, height, width, reverse=False, dtype=torch.float32, device=None, requires_grad=False):
    """Get a 3D checkerboard mask where no two adjacent entries have the same value.
    
    Args:
        depth (int): Number of depth slices.
        height (int): Number of rows.
        width (int): Number of columns.
        reverse (bool): If True, invert the mask.
        dtype (torch.dtype): Data type of the tensor.
        requires_grad (bool): Whether the tensor requires gradient.

    Returns:
        torch.Tensor: Mask of shape (1, 1, depth, height, width).
    """
    i = torch.arange(depth).reshape(-1, 1, 1)
    j = torch.arange(height).reshape(1, -1, 1)
    k = torch.arange(width).reshape(1, 1, -1)
    mask = (i + j + k) % 2
    mask = mask.to(dtype).unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(mask, dtype=dtype, device=device, requires_grad=requires_grad)
    if reverse:
        mask = 1 - mask
    return mask

class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        if self.mask_type == MaskType.CHANNEL_WISE:
            in_channels //= 2

        # Build scale and translate network
        growth_rate, steps = 64, 5
        self.st_net = nn.Sequential(
            DenseNetLayer3D(inplanes=in_channels,
                          growth_rate=growth_rate,
                          steps=steps),
            Conv3d(in_channels + growth_rate*steps, 2*in_channels, 
                    kernel_size=3, stride=1, padding=1)
        )

        # Learnable scale for s
        self.rescale = nn.utils.parametrizations.weight_norm(Rescale(in_channels))

    def forward(self, x, sldj=None, reverse=True):
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_mask_3d(x.size(2), x.size(3), x.size(4), self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_net(x_id)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x

class RealNVP(nn.Module):
    """RealNVP Model
    Codebase from Chris Chute:
    https://github.com/chrischute/real-nvp

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """
    def __init__(self, input_shape, mid_channels=64, num_blocks=5, num_scales=2, prior='std_normal'):
        super().__init__()
        self.flows = _RealNVP(0, num_scales, input_shape[0], mid_channels, num_blocks)

        # self.nbits = 8.
        if prior=='std_normal':
            self.prior = StandardNormal(input_shape)
        elif prior=='mog':
            self.prior = MixtureOfGaussians(input_shape)


    @torch.no_grad()
    def sample(self, z_shape, n_samples, device, **kwargs):
        """Sample from RealNVP model.
        Args:
            z_shape (tuple): 
            n_samples (int): Number of samples to generate.
            device (torch.device): Device to use.
        """
        z = self.prior.sample(n_samples).to(device)
        x, _ = self.forward(z, reverse=True)
        return x


    def log_p(self, x, **kwargs):
        """ returns the log likelihood.
        """
        z, sldj = self.forward(x, reverse=False)
        ll = (self.prior.log_p(z) + sldj)


        # prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        # prior_ll = prior_ll.flatten(1).sum(-1) - np.log(2**self.nbits) * np.prod(z.size()[1:])
        # ll = prior_ll + sldj
        # ll = ll.mean()

        return ll


    def forward(self, x, reverse=False):
        sldj = None
        if not reverse:
            sldj = 0    # we do not quintize !
            #  quintize !
            # x = (x * (2**self.nbits - 1) + torch.rand_like(x)) / (2**self.nbits)

        x, sldj = self.flows(x, sldj, reverse)
        return x, sldj

class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(8 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(8 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(8 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 4 * in_channels, 2 * mid_channels, num_blocks)


    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2x2(x, reverse=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2x2(x, reverse=True)

                # Re-squeeze -> split -> next block
                x = squeeze_2x2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2x2(x, reverse=True, alt_order=True)

        return x, sldj