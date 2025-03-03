from functools import partial

import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from srvae_backbone import *

def get_shape(z_dim):
    """ Given the dimentionality of the latent space,
        re-shape it to an appropriate 3-D tensor.
    """
    d = 8
    if (z_dim%d==0) and (z_dim // (d*d*d) > 0):  # cx8x8
        H = W = D = d
        C = z_dim // (d*d*d)
        return (C, H, W, D)
    raise "Latent space can not mapped to a 4-D tensor. \
            Please choose another dimentionality (power of 2)."

def log_normal_diag(z, z_mu, z_logvar):
    eps = 1e-12
    log_probs = z_logvar + (z - z_mu).pow(2).div(z_logvar.exp() + eps) + math.log(math.pi * 2.)
    log_probs = -0.5 * log_probs.view(z.size(0), -1).sum(dim=1)
    return log_probs

# ----- Two Staged VAE -----

class srVAE3D(nn.Module):
    """
    Super-Resolution Variational Auto-Encoder (srVAE).
    A Two Staged Visual Processing Variational AutoEncoder.

    Author:
    Ioannis Gatopoulos.
    """
    def __init__(self, x_shape, y_shape, u_dim, z_dim, prior):
        super().__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape

        self.u_shape = get_shape(u_dim)
        self.z_shape = get_shape(z_dim)
        # p(u)
        self.p_u = prior(self.u_shape)

        # q(u | y)
        self.q_u = q_u(self.u_shape, self.y_shape)

        # p(z | y)
        self.p_z = p_z(self.z_shape, (self.y_shape, self.u_shape))

        # q(z | x)
        self.q_z = q_z(self.z_shape, self.x_shape)

        # p(y | u)
        self.p_y = p_y(self.y_shape, self.u_shape)

        # p(x | y, z)
        self.p_x = p_x(self.x_shape, (self.y_shape, self.z_shape))

        # likelihood distribution
        self.recon_loss = nn.MSELoss(reduction='mean')

    @staticmethod
    def reparameterize(z_mean, z_log_var):
        """ z ~ N(z| z_mu, z_logvar) """
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var)*epsilon


    @torch.no_grad()
    def generate(self, n_samples=20):
        # u ~ p(u)
        u = self.p_u.sample(self.u_shape, n_samples=n_samples, device=self.device).to(self.device)

        # p(y|u)
        y_hat = self.p_y(u)

        # z ~ p(z|y, u)
        z_p_mean, z_p_logvar = self.p_z((y_hat, u))
        z_p = self.reparameterize(z_p_mean, z_p_logvar)

        # x ~ p(x|y,z)
        x_hat = self.p_x((y_hat, z_p))
        return x_hat, y_hat


    @torch.no_grad()
    def reconstruct(self, x, **kwargs):
        outputs = self.forward(x)
        return outputs.get('y'), outputs.get('y_hat'), outputs.get('x_hat')


    @torch.no_grad()
    def super_resolution(self, y):
        # u ~ q(u| y)
        u_q_mean, u_q_logvar = self.q_u(y)
        u_q = self.reparameterize(u_q_mean, u_q_logvar)

        # z ~ p(z|y)
        z_p_mean, z_p_logvar = self.p_z((y, u_q))
        z_p = self.reparameterize(z_p_mean, z_p_logvar)

        # x ~ p(x|y,z)
        x_hat = self.p_x((y, z_p))
        return x_hat


    def calculate_elbo(self, x, outputs, lag_weight, recon_weight, kl_weight, **kwargs):
        # unpack variables
        y, x_hat, y_hat = outputs.get('y'), outputs.get('x_hat'), outputs.get('y_hat')
        u_q, u_q_mean, u_q_logvar = outputs.get('u_q'), outputs.get('u_q_mean'), outputs.get('u_q_logvar')
        z_q, z_q_mean, z_q_logvar = outputs.get('z_q'), outputs.get('z_q_mean'), outputs.get('z_q_logvar')
        z_p_mean, z_p_logvar = outputs.get('z_p_mean'), outputs.get('z_p_logvar')
        # Reconstraction loss
        RE_x = self.recon_loss(x, x_hat)
        RE_y = self.recon_loss(y, y_hat)
        Re_xy = self.recon_loss(x_hat, y_hat)
        # Regularization loss
        log_p_u = self.p_u.log_p(u_q, dim=1)
        log_q_u = log_normal_diag(u_q, u_q_mean, u_q_logvar)
        KL_u = log_q_u - log_p_u
        KL_u = KL_u.mean()

        log_p_z = log_normal_diag(z_q, z_p_mean, z_p_logvar)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar)
        KL_z = log_q_z - log_p_z
        KL_z = KL_z.mean()

        # Total lower bound loss
        nelbo =  (lag_weight * (RE_x + RE_y + recon_weight * Re_xy ) + kl_weight * (KL_u + KL_z))

        diagnostics = {
            "nelbo" : nelbo.item(),

            "RE"    : (RE_x + RE_y + Re_xy).item(),
            "RE_x"  : RE_x.mean().item(),
            "RE_y"  : RE_y.mean().item(),
            "RE_xy" : Re_xy.mean().item(),
            "KL"    : (KL_z + KL_u).item(),
            "KL_u"  : KL_u.mean().item(),
            "KL_z"  : KL_z.mean().item(),
        }
        return nelbo, diagnostics


    def forward(self, x, y,**kwargs):
        """ Forward pass through the inference and the generative model. """
        # u ~ q(u| y)
        u_q_mean, u_q_logvar = self.q_u(y)
        u_q = self.reparameterize(u_q_mean, u_q_logvar)

        # z ~ q(z| x, y)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x ~ p(x| y, z)
        x_hat = self.p_x((y, z_q))

        # y ~ p(y| u)
        y_hat = self.p_y(u_q)

        # z ~ p(z| x)
        z_p_mean, z_p_logvar = self.p_z((y, u_q))
        return {
            'u_q_mean'   : u_q_mean,
            'u_q_logvar' : u_q_logvar,
            'u_q'        : u_q,

            'z_q_mean'   : z_q_mean,
            'z_q_logvar' : z_q_logvar,
            'z_q'        : z_q,

            'z_p_mean'   : z_p_mean,
            'z_p_logvar' : z_p_logvar,

            'y'          : y,
            'y_hat'   : y_hat,

            'x_hat'   : x_hat
        }


if __name__ == "__main__":
    pass