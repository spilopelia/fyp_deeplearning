import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from periodic_padding import periodic_padding_3d
from lag2eul import lag2eul
from magvit import MagVitAE3D, MagVitVAE3D
from srvae import srVAE3D
from srvae_prior import *
VAE_list = ["VAE", "VAEwithRes", "MagVitVAE3D"]

def crop_tensor(x):
	x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
	return x

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

import torch

def cyclical_annealing(step, steps_per_cycle, ratio=0.5):
    """
    Calculate the KL weight using a cyclical annealing schedule.

    Args:
        step (int): Current training step.
        steps_per_cycle (int): Number of steps in one complete cycle.
        ratio (float): Proportion of the cycle used for increasing the KL weight.

    Returns:
        torch.Tensor: KL weight (beta) for the current step.
    """
    cycle_position = step % steps_per_cycle
    cycle_progress = cycle_position / steps_per_cycle

    if cycle_progress < ratio:
        return 0.5 * (1 - torch.cos(torch.tensor(cycle_progress / ratio * torch.pi)))
    else:
        return torch.tensor(1.0)

# Assuming conv3x3 and BasicBlock are defined as in your original code.
class BasicBlock(nn.Module):
	def __init__(self,inplane,outplane,stride = 1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
		self.bn1 = nn.BatchNorm3d(outplane)
		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		return out

class ResBlock(nn.Module):
    def __init__(self,inplane,outplane,stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        orig_x = x
        x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out + orig_x
    
class Lpt2NbodyNet(nn.Module):
    def __init__(self, block, init_dim=4):
        super(Lpt2NbodyNet, self).__init__()
        self.layer1 = self._make_layer(block, init_dim, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(block, 64, 128, blocks=1, stride=2)
        self.layer3 = self._make_layer(block, 128, 128, blocks=2, stride=1)
        self.layer4 = self._make_layer(block, 128, 256, blocks=1, stride=2)
        self.layer5 = self._make_layer(block, 256, 256, blocks=2, stride=1)
        self.deconv1 = nn.ConvTranspose3d(256, 128, 3, stride=2, padding=0)
        self.deconv_batchnorm1 = nn.BatchNorm3d(num_features=128, momentum=0.1)
        self.layer6 = self._make_layer(block, 256, 128, blocks=2, stride=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, 3, stride=2, padding=0)
        self.deconv_batchnorm2 = nn.BatchNorm3d(num_features=64, momentum=0.1)
        self.layer7 = self._make_layer(block, 128, 64, blocks=2, stride=1)
        self.deconv4 = nn.ConvTranspose3d(64, init_dim, 1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)
        x = self.layer4(x2)
        x = self.layer5(x)
        x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))
        x = nn.functional.relu(self.deconv_batchnorm1(crop_tensor(self.deconv1(x))), inplace=True)
        x = torch.cat((x, x2), dim=1)
        x = self.layer6(x)
        x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))
        x = nn.functional.relu(self.deconv_batchnorm2(crop_tensor(self.deconv2(x))), inplace=True)
        x = torch.cat((x, x1), dim=1)
        x = self.layer7(x)
        x = self.deconv4(x)
        return x
    
class UNet3D(nn.Module):  
    def __init__(self, block, num_layers=2, base_filters=64, blocks_per_layer=2,init_dim=3):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Decoder path
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.ConvTranspose3d(out_channels, init_dim, 1, stride=1, padding=0)

        # Predefine BatchNorm3d and ReLU layers for each decoder step
        #self.batch_norms = nn.ModuleList()
        #self.relu = nn.ReLU(inplace=True)
        #for i in range(num_layers):
        #    self.batch_norms.insert(0, nn.BatchNorm3d(base_filters * (2 ** i)))  # Adjust channels accordingly

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        encoder_outputs = []

        x = self.init_conv(x)
        encoder_outputs.append(x)
        
        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)  # Compression layer
            x = self.encoders[i + 1](x)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))  # Assuming this is a custom function
            x = self.decoders[i](x)  # Transpose Conv layer
            x = crop_tensor(x)  # Assuming this is a custom function to crop the tensor
            
            # Use the pre-defined BatchNorm3d and ReLU layers
            #x = self.batch_norms[i // 2](x)  # BatchNorm
            #x = nn.ReLU(inplace=True)(x)  # ReLU
            
            # Skip connection with encoder outputs
            x = torch.cat((x, encoder_outputs[len(encoder_outputs)-2-i//2]), dim=1)  # Skip connection
            
            x = self.decoders[i + 1](x)  # Non-compression layer

        # Final 1x1 Conv
        x = self.final_conv(x)
        return x

class UNet3DwithRes(nn.Module):  
    def __init__(self, block, num_layers=2, base_filters=64, blocks_per_layer=2,init_dim=3):
        super(UNet3DwithRes, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Decoder path
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.ConvTranspose3d(out_channels, init_dim, 1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            if inplanes == outplanes:
                layers.append(ResBlock(inplanes, outplanes, stride=stride))
            else:
                layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        encoder_outputs = []

        x = self.init_conv(x)
        encoder_outputs.append(x)
        
        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)  # Compression layer
            x = self.encoders[i + 1](x)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))  # Assuming this is a custom function
            x = self.decoders[i](x)  # Transpose Conv layer
            x = crop_tensor(x)  # Assuming this is a custom function to crop the tensor
            
            x = torch.cat((x, encoder_outputs[len(encoder_outputs)-2-i//2]), dim=1)  # Skip connection
            
            x = self.decoders[i + 1](x)  # Non-compression layer

        # Final 1x1 Conv
        x = self.final_conv(x)
        return x

class VAE3D(nn.Module):
    def __init__(self, block, num_layers=2, base_filters=64, blocks_per_layer=2, 
                 init_dim=3, latent_dim=32):
        super(VAE3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, 
                                        blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, 
                                                blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, 
                                                blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Variational bottleneck
        self.conv_mu = nn.Conv3d(out_channels, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv3d(out_channels, latent_dim, kernel_size=1)
        self.conv_decode = nn.Conv3d(latent_dim, out_channels, kernel_size=1)

        # Decoder path (no skip connections)
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, 
                                                  kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels//2, out_channels//2,
                                                blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.Conv3d(out_channels, init_dim, kernel_size=1)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)
            x = self.encoders[i+1](x)

        # Variational bottleneck
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.conv_decode(z)

        # Decoder
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))  # Your custom padding
            x = self.decoders[i](x)  # Transposed convolution
            x = crop_tensor(x)  # Your custom cropping
            x = self.decoders[i+1](x)  # Regular convolution block

        x = self.final_conv(x)
        return x, mu, logvar

class VAE3DwithRes(nn.Module):
    def __init__(self, block, num_layers=2, base_filters=64, blocks_per_layer=2, 
                 init_dim=3, latent_dim=32):
        super(VAE3DwithRes, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, 
                                        blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, 
                                                blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, 
                                                blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Variational bottleneck
        self.conv_mu = nn.Conv3d(out_channels, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv3d(out_channels, latent_dim, kernel_size=1)
        self.conv_decode = nn.Conv3d(latent_dim, out_channels, kernel_size=1)

        # Decoder path (no skip connections)
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, 
                                                  kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels//2, out_channels//2,
                                                blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.Conv3d(out_channels, init_dim, kernel_size=1)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            if inplanes == outplanes:
                layers.append(ResBlock(inplanes, outplanes, stride=stride))
            else:
                layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)
            x = self.encoders[i+1](x)

        # Variational bottleneck
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.conv_decode(z)

        # Decoder
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))  # Your custom padding
            x = self.decoders[i](x)  # Transposed convolution
            x = crop_tensor(x)  # Your custom cropping
            x = self.decoders[i+1](x)  # Regular convolution block

        x = self.final_conv(x)
        return x, mu, logvar

class AE3DwithRes(nn.Module):
    def __init__(self, block, num_layers=2, base_filters=64, blocks_per_layer=2, 
                 init_dim=3, latent_dim=32):
        super(AE3DwithRes, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, 
                                        blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, 
                                                blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, 
                                                blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Variational bottleneck
        self.conv_encode = nn.Conv3d(out_channels, latent_dim, kernel_size=1)
        self.conv_decode = nn.Conv3d(latent_dim, out_channels, kernel_size=1)

        # Decoder path (no skip connections)
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, 
                                                  kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels//2, out_channels//2,
                                                blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.Conv3d(out_channels, init_dim, kernel_size=1)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            if inplanes == outplanes:
                layers.append(ResBlock(inplanes, outplanes, stride=stride))
            else:
                layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)



    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)
            x = self.encoders[i+1](x)

        # Variational bottleneck
        z = self.conv_encode(x)

        x = self.conv_decode(z)

        # Decoder
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))  # Your custom padding
            x = self.decoders[i](x)  # Transposed convolution
            x = crop_tensor(x)  # Your custom cropping
            x = self.decoders[i+1](x)  # Regular convolution block

        x = self.final_conv(x)
        return x
    
# LightningModule wrapping the Lpt2NbodyNet
class Lpt2NbodyNetLightning(pl.LightningModule):
    def __init__(self, 
                lr: float = 1.0e-4,
                beta1: float = 0.9,
                beta2: float = 0.999,
                weight_decay: float = 1.0e-4,  
                optimizer: str = 'Adam',
                lr_scheduler: str = 'Constant',
                lr_warmup: int = 1000,
                num_samples: int = 30000,
                batch_size: int = 128,
                max_epochs: int = 500,
                model: str = 'default',
                num_layers: int = 4, 
                base_filters: int = 64, 
                blocks_per_layer: int = 2, 
                init_dim: int = 3,
                latent_dim: int = 32,
                reversed: bool = False,
                normalized: bool = False,
                normalized_scale: float = 128.0,
                eul_loss: bool = False,
                kl_loss: bool = True,
                kl_loss_annealing: bool = True,
                steps_per_cycle: int = 1000,
                kl_ratio: float = 0.5,
                eul_loss_scale: float = 1.0,
                lag_loss_scale: float = 1.0,
                recon_loss_scale: float = 0.0,                
                kl_loss_scale: float = 1.0,
                srvae_udim: int = 1024,
                srvae_zdim: int = 1024,
                srvae_prior: str = 'MixtureOfGaussians',
                **kwargs
                ):
        super(Lpt2NbodyNetLightning, self).__init__()

        self.save_hyperparameters(ignore=['kwargs'])  # This will save all init args except kwargs
        self.model_type = model

        if model == "default":
            self.model = Lpt2NbodyNet(BasicBlock,init_dim=self.hparams.init_dim)
        elif model == "UNet":
            self.model = UNet3D(block=BasicBlock,num_layers=self.hparams.num_layers,
                                base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer,init_dim=self.hparams.init_dim)
        elif model == "UNet3DwithRes":
            self.model = UNet3DwithRes(block=BasicBlock,num_layers=self.hparams.num_layers,
                                base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer,init_dim=self.hparams.init_dim)
        elif model == "VAE":
            self.model = VAE3D(block=BasicBlock, num_layers=self.hparams.num_layers, base_filters=self.hparams.base_filters, 
                                blocks_per_layer=self.hparams.blocks_per_layer, init_dim=self.hparams.init_dim, latent_dim=self.hparams.latent_dim)
        elif model == "VAEwithRes":
            self.model = VAE3DwithRes(block=BasicBlock, num_layers=self.hparams.num_layers, base_filters=self.hparams.base_filters, 
                                blocks_per_layer=self.hparams.blocks_per_layer, init_dim=self.hparams.init_dim, latent_dim=self.hparams.latent_dim)
        elif model == "AEwithRes":
            self.model = AE3DwithRes(block=BasicBlock, num_layers=self.hparams.num_layers, base_filters=self.hparams.base_filters, 
                                blocks_per_layer=self.hparams.blocks_per_layer, init_dim=self.hparams.init_dim, latent_dim=self.hparams.latent_dim)
        elif model == "MagVitAE3D":
            self.model = MagVitAE3D(n_bands = self.hparams.init_dim, hidden_dims = self.hparams.base_filters, residual_conv_kernel_size = 3,
                                n_compressions = self.hparams.num_layers, num_consecutive = self.hparams.blocks_per_layer,)
        elif model == "MagVitVAE3D":
            self.model = MagVitVAE3D(n_bands = self.hparams.init_dim, hidden_dims = self.hparams.base_filters, residual_conv_kernel_size = 3,
                                n_compressions = self.hparams.num_layers, num_consecutive = self.hparams.blocks_per_layer,) 
        elif model == "SRVAE3D":
            self.model = srVAE3D(x_shape = (3, 32, 32, 32), y_shape = (3, 32, 32, 32), 
                                u_dim = self.hparams.srvae_udim, z_dim = self.hparams.srvae_zdim, prior=self.hparams.srvae_prior)
        self.criterion = nn.MSELoss()  

    def forward(self, x):
        return self.model(x)

    #def normalize(self, x, scale):
    #    return x / scale
    
    #def denormalize(self, x, scale):
    #    return x * scale
    
    def training_step(self, batch, batch_idx):
        # Reverse batch if needed
        x, y = batch if not self.hparams.reversed else (batch[1], batch[0])

        #if self.hparams.normalized:
        #    x = self.normalize(x, self.hparams.normalized_scale)
        #    y = self.normalize(y, self.hparams.normalized_scale)

        # Forward pass
        if self.model_type in VAE_list:
            y_hat, mu, logvar = self(x)
        elif self.model_type == "SRVAE3D":
            output = self(x)
            if self.hparams.kl_loss_annealing:
                # Calculate current training step
                current_step = self.global_step

                # Define steps per cycle
                steps_per_cycle = self.hparams.steps_per_cycle

                # Compute KL weight using cyclical annealing
                kl_weight_ratio = cyclical_annealing(current_step, steps_per_cycle,
                                               ratio=self.hparams.kl_ratio).to(self.device)
                kl_weight = self.hparams.kl_loss_scale * kl_weight_ratio
                self.log('kl_weight', kl_weight, on_step=True, on_epoch=False, logger=True, sync_dist=True)

            loss, diagnostics = srVAE3D.calculate_elbo(x, output, self.hparams.lag_loss_scale, self.hparams.recon_loss_scale, kl_weight)
            self.log('train_batch_loss', diagnostics['nelbo'], on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('train_epoch_loss', diagnostics['nelbo'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_lag_loss', diagnostics['Re_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_lag_loss_x', diagnostics['Re_x'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_lag_loss_y', diagnostics['Re_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_kl_loss', diagnostics['KL'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_kl_loss_u', diagnostics['KL_u'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_kl_loss_z', diagnostics['KL_z'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            return loss
        else:
            y_hat = self(x)
            
        # Base lagrangian loss
        lag_loss = self.criterion(y_hat, y)
        train_loss = lag_loss

        # Flags for loss conditions
        eul_enabled = self.hparams.eul_loss
        kl_enabled = (self.model_type in VAE_list) and self.hparams.kl_loss

        # Apply lagrangian scaling if either auxiliary loss is active
        if eul_enabled or kl_enabled:
            train_loss = lag_loss * self.hparams.lag_loss_scale

        # Euler loss component
        if eul_enabled:
            eul_y_hat, eul_y = lag2eul([y_hat, y])
            eul_loss = self.criterion(eul_y_hat, eul_y)
            train_loss += eul_loss * self.hparams.eul_loss_scale

        # KL divergence component
        if kl_enabled:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1) 
            kl_loss = kl_loss.mean()
            kl_weight = 1.0
            if self.hparams.kl_loss_annealing:
                # Calculate current training step
                current_step = self.global_step

                # Define steps per cycle
                steps_per_cycle = self.hparams.steps_per_cycle

                # Compute KL weight using cyclical annealing
                kl_weight_ratio = cyclical_annealing(current_step, steps_per_cycle,
                                               ratio=self.hparams.kl_ratio).to(self.device)
                kl_weight = self.hparams.kl_loss_scale * kl_weight_ratio
                self.log('kl_weight', kl_weight, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            # Apply the annealed KL weight
            train_loss += kl_loss * kl_weight 

        # Logging
        self.log('train_batch_loss', train_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.log('train_epoch_loss', train_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_epoch_lag_loss', lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if eul_enabled:
            self.log('train_epoch_eul_loss', eul_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if kl_enabled:
            self.log('train_epoch_kl_loss', kl_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # Reverse batch if needed
        x, y = batch if not self.hparams.reversed else (batch[1], batch[0])

        if self.hparams.normalized:
            x = self.normalize(x, self.hparams.normalized_scale)
            y = self.normalize(y, self.hparams.normalized_scale)
        # Forward pass
        if self.model_type in VAE_list:
            y_hat, mu, logvar = self(x)
        elif self.model_type == "SRVAE3D":
            output = self(x)
            loss, diagnostics = srVAE3D.calculate_elbo(x, output, self.hparams.lag_loss_scale, self.hparams.recon_loss_scale)
            self.log('val_batch_loss', diagnostics['nelbo'], on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('val_epoch_loss', diagnostics['nelbo'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_lag_loss', diagnostics['Re_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_lag_loss_x', diagnostics['Re_x'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_lag_loss_y', diagnostics['Re_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_kl_loss', diagnostics['KL'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_kl_loss_u', diagnostics['KL_u'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_kl_loss_z', diagnostics['KL_z'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            return loss
        else:
            y_hat = self(x)
            
        # Base lagrangian loss
        lag_loss = self.criterion(y_hat, y)
        val_loss = lag_loss

        # Flags for loss conditions
        eul_enabled = self.hparams.eul_loss
        kl_enabled = (self.model_type in VAE_list) and self.hparams.kl_loss

        # Apply lagrangian scaling if either auxiliary loss is active
        if eul_enabled or kl_enabled:
            val_loss = lag_loss * self.hparams.lag_loss_scale

        # Euler loss component
        if eul_enabled:
            eul_y_hat, eul_y = lag2eul([y_hat, y])
            eul_loss = self.criterion(eul_y_hat, eul_y)
            val_loss += eul_loss * self.hparams.eul_loss_scale

        # KL divergence component
        if kl_enabled:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1) 
            kl_loss = kl_loss.mean()
            val_loss += kl_loss * self.hparams.kl_loss_scale

        # Logging
        self.log('val_epoch_loss', val_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_epoch_lag_loss', lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if eul_enabled:
            self.log('val_epoch_eul_loss', eul_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if kl_enabled:
            self.log('val_epoch_kl_loss', kl_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if self.hparams.normalized:
            y_hat = self.denormalize(y_hat, self.hparams.normalized_scale)
        return y_hat
    
    def test_step(self, batch, batch_idx):
        # Reverse batch if needed
        x, y = batch if not self.hparams.reversed else (batch[1], batch[0])

        if self.hparams.normalized:
            x = self.normalize(x, self.hparams.normalized_scale)
            y = self.normalize(y, self.hparams.normalized_scale)
        # Forward pass
        if self.model_type in VAE_list:
            y_hat, mu, logvar = self(x)
        elif self.model_type == "SRVAE3D":
            output = self(x)
            loss, diagnostics = srVAE3D.calculate_elbo(x, output, self.hparams.lag_loss_scale, self.hparams.recon_loss_scale)
            self.log('test_batch_loss', diagnostics['nelbo'], on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('test_epoch_loss', diagnostics['nelbo'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('test_epoch_lag_loss', diagnostics['Re_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('test_epoch_lag_loss_x', diagnostics['Re_x'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('test_epoch_lag_loss_y', diagnostics['Re_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('test_epoch_kl_loss', diagnostics['KL'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('test_epoch_kl_loss_u', diagnostics['KL_u'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('test_epoch_kl_loss_z', diagnostics['KL_z'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            return loss
        else:
            y_hat = self(x)
            
        # Base lagrangian loss
        lag_loss = self.criterion(y_hat, y)
        test_loss = lag_loss

        # Flags for loss conditions
        eul_enabled = self.hparams.eul_loss
        kl_enabled = (self.model_type in VAE_list) and self.hparams.kl_loss

        # Apply lagrangian scaling if either auxiliary loss is active
        if eul_enabled or kl_enabled:
            test_loss = lag_loss * self.hparams.lag_loss_scale

        # Euler loss component
        if eul_enabled:
            eul_y_hat, eul_y = lag2eul([y_hat, y])
            eul_loss = self.criterion(eul_y_hat, eul_y)
            test_loss += eul_loss * self.hparams.eul_loss_scale

        # KL divergence component
        if kl_enabled:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            test_loss += kl_loss * self.hparams.kl_loss_scale

        if self.hparams.normalized:
            y_hat = self.denormalize(y_hat, self.hparams.normalized_scale)
        return y_hat
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        else:
            optimizer = optim.Adam(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_scheduler == 'Constant':
            return optimizer

        elif self.hparams.lr_scheduler == 'Cosine':
            total_steps = self.hparams.max_epochs * (self.hparams.num_samples // self.hparams.batch_size)
            T_max = total_steps - self.hparams.lr_warmup
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.hparams.lr,
                    total_iters=self.hparams.lr_warmup,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max
                ),
            ],
            milestones=[self.hparams.lr_warmup],
        )
            scheduler = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": True,
            }
            return [optimizer], [scheduler]

        return optimizer