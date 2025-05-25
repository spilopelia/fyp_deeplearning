import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from lag2eul import lag2eul
from model_backbone import Lpt2NbodyNet, UNet3D, UNet3DwithRes, VAE3D, VAE3DwithRes, AE3DwithRes, BasicBlock
from magvit import MagVitAE3D, MagVitVAE3D
from srvae import srVAE3D
from srvae_prior import *
from score_model import UNet3DModel
from score_vesde import VESDE, get_sigma_time, get_sample_time
from torch_ema import ExponentialMovingAverage
from ddim_sampler import DDIMSampler, DDPMSampler, extract
from ddim_model import ddim_UNet3D
from naf_net import NAFNet3D, BaselineBlock, BaselineBlock_SCA, BaselineBlock_SG, NAFBlock3D
from torchmetrics.image import PeakSignalNoiseRatio
torch.backends.cudnn.benchmark = True
VAE_list = ["VAE", "VAEwithRes", "MagVitVAE3D"]
    
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
                lr_cosine_period = None,
                num_samples: int = 30000,
                batch_size: int = 128,
                max_epochs: int = 500,
                used_density: bool = False,
                model: str = 'default',
                num_layers: int = 4, 
                base_filters: int = 64, 
                blocks_per_layer: int = 2, 
                init_dim: int = 3,
                latent_dim: int = 32,
                reversed: bool = False,
                normalized: bool = False,
                normalized_scale: float = None,
                standardized: bool = False,
                standardized_mean_1: float = None,
                standardized_std_1: float = None,
                standardized_mean_2: float = None,
                standardized_std_2: float = None,
                compressed: bool = False,
                compression_type: str = 'arcsinh',
                compression_factor: float = 24,
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
                diffusion_noise_sigma: float = 0.1,
                diffusion_sigma_min: float = 0.01,
                diffusion_sigma_max: float = 100.0,
                diffusion_sampling_eps: float = 1.0e-5,
                diffusion_T: float = 1.0,
                diffusion_num_scales: int = 1000,
                score_act_f: str ='swish',  
                ch_mult = [1, 2, 2, 1, 1],  
                dropout: float = 0.1,  
                num_input_channels: int = 6,  
                num_output_channels: int = 3 ,  
                ema: bool = False,
                ema_rate: float = 0.999,
                ddim_beta = [0.0001, 0.02],
                sampler: str = 'DDIM',
                naf_middle_blk_num = 12,
                naf_enc_blk_nums = [2, 2, 4, 8],
                naf_dec_blk_nums = [2, 2, 2, 2],
                naf_dw_expand = 2,
                naf_ffn_expand = 2,
                **kwargs
                ):
        super(Lpt2NbodyNetLightning, self).__init__()

        self.save_hyperparameters(ignore=['kwargs'])  # This will save all init args except kwargs
        self.model_type = model
        if not used_density:
            self.train_epoch_lag_loss = 'train_epoch_lag_loss'
            self.val_epoch_lag_loss = 'val_epoch_lag_loss'
        else:
            self.train_epoch_lag_loss = 'train_epoch_eul_loss'
            self.val_epoch_lag_loss = 'val_epoch_eul_loss'
        if reversed:
            standardized_mean_1, standardized_mean_2 = standardized_mean_2, standardized_mean_1
            standardized_std_1, standardized_std_2 = standardized_std_2, standardized_std_1

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
            if self.hparams.srvae_prior == 'MixtureOfGaussians':
                self.model = srVAE3D(x_shape = (3, 32, 32, 32), y_shape = (3, 32, 32, 32), 
                                u_dim = self.hparams.srvae_udim, z_dim = self.hparams.srvae_zdim, prior=MixtureOfGaussians)
            elif self.hparams.srvae_prior == 'StandardNormal':
                self.model = srVAE3D(x_shape = (3, 32, 32, 32), y_shape = (3, 32, 32, 32), 
                            u_dim = self.hparams.srvae_udim, z_dim = self.hparams.srvae_zdim, prior=StandardNormal)
            elif self.hparams.srvae_prior == 'RealNVP':
                self.model = srVAE3D(x_shape = (3, 32, 32, 32), y_shape = (3, 32, 32, 32), 
                            u_dim = self.hparams.srvae_udim, z_dim = self.hparams.srvae_zdim, prior=RealNVP)
        elif model == "NAFNet3D_base":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = BaselineBlock)
        elif model == "NAFNet3D_base_SG":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = BaselineBlock_SG)
        elif model == "NAFNet3D_base_SCA":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = BaselineBlock_SCA)
        elif model == "NAFNet3D":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = NAFBlock3D)
        elif model == "ICdiffusion":
            self.model = UNet3DModel(
                    act_f=score_act_f,  # From "model.nonlinearity": "swish"
                    nf=base_filters,  # From "model.nf": 32
                    ch_mult=ch_mult,  # From "model.ch_mult": [1, 2, 2, 1, 1]
                    num_res_blocks=blocks_per_layer,  # From "model.num_res_blocks": 2
                    dropout=dropout,  # From "model.dropout": 0.1
                    resamp_with_conv=True,  # From "model.resamp_with_conv": true
                    image_size=32,  # From "data.image_size": 128
                    conditional=True,  # From "model.conditional": true
                    fir=False,  # From "model.fir": false
                    fir_kernel=[1, 3, 3, 1],  # From "model.fir_kernel": [1, 3, 3, 1]
                    skip_rescale=True,  # From "model.skip_rescale": true
                    embedding_type="fourier",  # From "model.embedding_type": "fourier"
                    init_scale=0.0,  # From "model.init_scale": 0.0
                    num_input_channels=num_input_channels,  # From "data.num_input_channels": 2
                    num_output_channels=num_output_channels,  # From "data.num_output_channels": 1
                    fourier_scale=16,  # From "model.fourier_scale": 16
                )
            self.sde = VESDE(self.hparams.diffusion_sigma_min, self.hparams.diffusion_sigma_max, self.hparams.diffusion_num_scales, self.hparams.diffusion_T, self.hparams.diffusion_sampling_eps)
        elif model == "DDIM":
            self.model = ddim_UNet3D(
                    in_channels=num_input_channels,         # adjust for volumetric data (e.g. 1 for grayscale, 3 for RGB)
                    model_channels=base_filters,
                    out_channels=num_output_channels,
                    num_res_blocks=blocks_per_layer,
                    attention_resolutions=(8, 16),
                    dropout=dropout,
                    channel_mult=tuple(ch_mult),
                    conv_resample=True,
                    num_heads=4
                )
            # generate T steps of beta
            self.register_buffer("beta_t", torch.linspace(*ddim_beta, diffusion_num_scales, dtype=torch.float32))

            # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
            alpha_t = 1.0 - self.beta_t
            alpha_t_bar = torch.cumprod(alpha_t, dim=0)

            # calculate and store two coefficient of $q(x_t | x_0)$
            self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
            self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

        self.criterion = nn.MSELoss()  
        self.psnr = PeakSignalNoiseRatio(data_range=128)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(x,y)
        return self.model(x)

    def on_fit_start(self):
        if self.hparams.ema == True:
            self.ema = ExponentialMovingAverage(self.model.parameters(), self.hparams.ema_rate)

    def training_step(self, batch, batch_idx):
        # Reverse batch if needed
        x, y = batch if not self.hparams.reversed else (batch[1], batch[0])

        if self.hparams.normalized:
            x, x_scale = self.normalize_displacement_field(x, self.hparams.normalized_scale)
            y, y_scale = self.normalize_displacement_field(y, self.hparams.normalized_scale)

        if self.hparams.standardized:
            x, x_mean, x_std = self.standardize_displacement_field(x, self.hparams.standardized_mean_1, self.hparams.standardized_std_1)
            y, y_mean, y_std = self.standardize_displacement_field(y, self.hparams.standardized_mean_2, self.hparams.standardized_std_2)
        
        if self.hparams.compressed:
            x = self.range_compression(x, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
            y = self.range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

        # Forward pass
        if self.model_type in VAE_list:
            y_hat, mu, logvar = self(x)

        elif self.model_type == "SRVAE3D":
            output = self(x,y)
            y_hat = output.get('y_hat')

            if self.hparams.normalized:
                x = x * x_scale
                y = y * y_scale
                y_hat = y_hat * y_scale

            if self.hparams.standardized:
                x = x * x_std + x_mean
                y = y * y_std + y_mean
                y_hat = y_hat * y_std + y_mean

            if self.hparams.compressed:
                x = self.reverse_range_compression(x, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                y_hat = self.reverse_range_compression(y_hat, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

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

                loss, diagnostics = self.model.calculate_elbo(x, output, self.hparams.lag_loss_scale, self.hparams.recon_loss_scale, kl_weight)
            loss, diagnostics = self.model.calculate_elbo(x, output, self.hparams.lag_loss_scale, self.hparams.recon_loss_scale, self.hparams.kl_loss_scale)

            self.log('train_batch_loss', diagnostics['nelbo'], on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('train_epoch_loss', diagnostics['nelbo'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_lag_loss', diagnostics['RE_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_lag_loss_x', diagnostics['RE_x'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_lag_loss_y', diagnostics['RE_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_kl_loss', diagnostics['KL'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_kl_loss_u', diagnostics['KL_u'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('train_epoch_kl_loss_z', diagnostics['KL_z'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            return loss

        elif self.model_type == "ICdiffusion":
            sigma_time = get_sigma_time(self.hparams.diffusion_sigma_min, self.hparams.diffusion_sigma_max)
            sample_time = get_sample_time(self.hparams.diffusion_sampling_eps, self.hparams.diffusion_T)

            B = y.size(dim=0)     
            #x += self.hparams.diffusion_noise_sigma * torch.randn_like(x).to(x.device)
            time_steps = sample_time(shape=(B,)).to(x.device)
            sigmas = sigma_time(time_steps).to(x.device)
            sigmas = sigmas[:,None,None,None,None]
            z = torch.randn_like(y,  device=y.device)
            inputs = torch.cat([y + sigmas * z, x], dim=1)  
            output = self.model(inputs, time_steps)
            loss = torch.sum(torch.square(output + z)) /  B
            self.log('train_batch_loss', loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('train_epoch_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            return loss

        elif self.model_type == "DDIM":
            # get a random training step $t \sim Uniform({1, ..., T})$
            t = torch.randint(self.hparams.diffusion_num_scales, size=(y.shape[0],), device=y.device)

            # generate $\epsilon \sim N(0, 1)$
            epsilon = torch.randn_like(y)

            # predict the noise added from $x_{t-1}$ to $x_t$
            y_t = (extract(self.signal_rate, t, y.shape) * y +
                extract(self.noise_rate, t, y.shape) * epsilon)
            inputs = torch.cat([y_t, x], dim=1)
            epsilon_theta = self.model(inputs, t)

            # get the gradient
            loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
            loss = torch.sum(loss)
            self.log('train_batch_loss', loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('train_epoch_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            return loss

        else:
            y_hat = self(x)

            if self.hparams.normalized:
                y = y * y_scale
                y_hat = y_hat * y_scale

            if self.hparams.standardized:
                y = y * y_std + y_mean
                y_hat = y_hat * y_std + y_mean

            if self.hparams.compressed:
                y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                y_hat = self.reverse_range_compression(y_hat, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

            # Base lagrangian loss
            lag_loss = self.criterion(y_hat, y)
            train_loss = lag_loss

            # Flags for loss conditions
            eul_enabled = self.hparams.eul_loss
            kl_enabled = (self.model_type in VAE_list) and self.hparams.kl_loss

            # Apply lagrangian scaling if either auxiliary loss is active
            if kl_enabled:
                train_loss = lag_loss * self.hparams.lag_loss_scale

            if eul_enabled:
                train_loss = torch.log(lag_loss) * self.hparams.lag_loss_scale

            # Euler loss component
            if eul_enabled:
                eul_y_hat, eul_y = lag2eul([y_hat, y])
                eul_loss = self.criterion(eul_y_hat, eul_y)
                train_loss += torch.log(eul_loss) * self.hparams.eul_loss_scale

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
            train_psnr = self.psnr(y_hat, y)
            self.log('train_batch_psnr', train_psnr, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('train_epoch_psnr', train_psnr, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            self.log('train_batch_loss', train_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('train_epoch_loss', train_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log(self.train_epoch_lag_loss, lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
            if eul_enabled:
                self.log('train_epoch_eul_loss', eul_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
            if kl_enabled:
                self.log('train_epoch_kl_loss', kl_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            return train_loss

    def validation_step(self, batch, batch_idx):
        # Reverse batch if needed
        x, y = batch if not self.hparams.reversed else (batch[1], batch[0])

        if self.hparams.normalized:
            x, x_scale = self.normalize_displacement_field(x, self.hparams.normalized_scale)
            y, y_scale = self.normalize_displacement_field(y, self.hparams.normalized_scale)
            
        if self.hparams.standardized:
            x, x_mean, x_std = self.standardize_displacement_field(x, self.hparams.standardized_mean_1, self.hparams.standardized_std_1)
            y, y_mean, y_std = self.standardize_displacement_field(y, self.hparams.standardized_mean_2, self.hparams.standardized_std_2)
        
        if self.hparams.compressed:
            x = self.range_compression(x, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
            y = self.range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
            
        # Forward pass
        if self.model_type in VAE_list:
            y_hat, mu, logvar = self(x)

        elif self.model_type == "SRVAE3D":
            output = self(x,y)
            y_hat = output.get('y_hat')

            if self.hparams.normalized:
                x = x * x_scale
                y = y * y_scale
                y_hat = y_hat * y_scale

            if self.hparams.standardized:
                x = x * x_std + x_mean
                y = y * y_std + y_mean
                y_hat = y_hat * y_std + y_mean

            if self.hparams.compressed:
                x = self.reverse_range_compression(x, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                y_hat = self.reverse_range_compression(y_hat, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

            loss, diagnostics = self.model.calculate_elbo(x, output, self.hparams.lag_loss_scale, self.hparams.recon_loss_scale, self.hparams.kl_loss_scale)

            self.log('val_batch_loss', diagnostics['nelbo'], on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('val_epoch_loss', diagnostics['nelbo'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_lag_loss', diagnostics['RE_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_lag_loss_x', diagnostics['RE_x'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_lag_loss_y', diagnostics['RE_xy'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_kl_loss', diagnostics['KL'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_kl_loss_u', diagnostics['KL_u'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_epoch_kl_loss_z', diagnostics['KL_z'], on_step=False, on_epoch=True, logger=True, sync_dist=True)

            return output.get('y_hat')

        elif self.model_type == "ICdiffusion":
            sigma_time = get_sigma_time(self.hparams.diffusion_sigma_min, self.hparams.diffusion_sigma_max)
            sample_time = get_sample_time(self.hparams.diffusion_sampling_eps, self.hparams.diffusion_T)

            B = y.size(dim=0)     
            #x += self.hparams.diffusion_noise_sigma * torch.randn_like(x).to(x.device)
            time_steps = sample_time(shape=(B,)).to(x.device)
            sigmas = sigma_time(time_steps).to(x.device)
            sigmas = sigmas[:,None,None,None,None]
            z = torch.randn_like(y,  device=y.device)
            inputs = torch.cat([y + sigmas * z, x], dim=1)  
            output = self.model(inputs, time_steps)
            loss = torch.sum(torch.square(output + z)) /  B
            self.log('val_batch_loss', loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('val_epoch_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            if self.trainer.current_epoch % 1 == 0 and batch_idx == 0:
                x = x[0:1]
                y = y[0:1]
                _, sampled_ic = self.sample_initial_condition(x)

                if self.hparams.normalized:
                    y = y * y_scale
                    sampled_ic = sampled_ic * y_scale

                if self.hparams.standardized:
                    y = y * y_std + y_mean
                    sampled_ic = sampled_ic * y_std + y_mean

                if self.hparams.compressed:
                    y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                    sampled_ic = self.reverse_range_compression(sampled_ic, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

                lag_loss = self.criterion(sampled_ic, y)
                val_psnr = self.psnr(sampled_ic, y)
                self.log('val_epoch_psnr', val_psnr, on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(self.val_epoch_lag_loss, lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
                return sampled_ic
            return None
        elif self.model_type == "DDIM":
            # get a random training step $t \sim Uniform({1, ..., T})$
            t = torch.randint(self.hparams.diffusion_num_scales, size=(y.shape[0],), device=y.device)

            # generate $\epsilon \sim N(0, 1)$
            epsilon = torch.randn_like(y)

            # predict the noise added from $x_{t-1}$ to $x_t$
            y_t = (extract(self.signal_rate, t, y.shape) * y +
                extract(self.noise_rate, t, y.shape) * epsilon)
            inputs = torch.cat([y_t, x], dim=1)
            epsilon_theta = self.model(inputs, t)

            # get the gradient
            loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
            loss = torch.sum(loss)
            self.log('val_epoch_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            if self.trainer.current_epoch % 1 == 0 and batch_idx == 0:
                x = x[0:1]
                y = y[0:1]
                sampled_ic = self.sample_initial_condition(x, self.hparams.sampler)

                if self.hparams.normalized:
                    y = y * y_scale
                    sampled_ic = sampled_ic * y_scale

                if self.hparams.standardized:
                    y = y * y_std + y_mean
                    sampled_ic = sampled_ic * y_std + y_mean

                if self.hparams.compressed:
                    y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                    sampled_ic = self.reverse_range_compression(sampled_ic, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

                lag_loss = self.criterion(sampled_ic, y)
                val_psnr = self.psnr(sampled_ic, y)
                self.log('val_epoch_psnr', val_psnr, on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(self.val_epoch_lag_loss, lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
                return sampled_ic
            return None            
        else:
            y_hat = self(x)

            if self.hparams.normalized:
                y = y * y_scale
                y_hat = y_hat * y_scale

            if self.hparams.standardized:
                y = y * y_std + y_mean
                y_hat = y_hat * y_std + y_mean

            if self.hparams.compressed:
                y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
                y_hat = self.reverse_range_compression(y_hat, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

            # Base lagrangian loss
            lag_loss = self.criterion(y_hat, y)
            val_loss = lag_loss

            # Flags for loss conditions
            eul_enabled = self.hparams.eul_loss
            kl_enabled = (self.model_type in VAE_list) and self.hparams.kl_loss

            # Apply lagrangian scaling if either auxiliary loss is active
            if kl_enabled:
                val_loss = lag_loss * self.hparams.lag_loss_scale

            if eul_enabled:
                val_loss = torch.log(lag_loss) * self.hparams.lag_loss_scale

            # Euler loss component
            if eul_enabled:
                eul_y_hat, eul_y = lag2eul([y_hat, y])
                eul_loss = self.criterion(eul_y_hat, eul_y)
                val_loss += torch.log(eul_loss) * self.hparams.eul_loss_scale

            # KL divergence component
            if kl_enabled:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1) 
                kl_loss = kl_loss.mean()
                val_loss += kl_loss * self.hparams.kl_loss_scale

            # Logging
            val_psnr = self.psnr(y_hat, y)
            self.log('val_batch_psnr', val_psnr, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('val_epoch_psnr', val_psnr, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            self.log('val_epoch_loss', val_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log(self.val_epoch_lag_loss, lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
            if batch_idx == 0 and not self.hparams.used_density:
                eul_y_hat, eul_y = lag2eul([y_hat, y])
                eul_loss = self.criterion(eul_y_hat, eul_y)    
                self.log('val_batch_eul_loss', eul_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True) 

            if eul_enabled:
                self.log('val_epoch_eul_loss', eul_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
            if kl_enabled:
                self.log('val_epoch_kl_loss', kl_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            return y_hat

    def on_before_zero_grad(self, *args, **kwargs):
        # Update EMA after each training step, post-optimization
        if self.hparams.ema == True:
            self.ema.update()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer == 'Adamax':
            optimizer = optim.Adamax(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_scheduler == 'Constant':
            return optimizer

        elif self.hparams.lr_scheduler == 'Cosine':
            if self.hparams.lr_cosine_period == None:
                total_steps = self.hparams.max_epochs * (self.hparams.num_samples // self.hparams.batch_size)
            else:
                total_steps = self.hparams.lr_cosine_period
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

    def normalize_displacement_field(self, displacement, scale=None):
        """
        Normalize the 5D displacement tensor [B, C, H, W, D] such that 
        the maximum displacement magnitude across the entire tensor becomes 1.
        This operation multiplies each displacement vector by the same factor,
        thereby preserving its direction.
        
        Returns:
        normalized: the normalized tensor
        scale: the global scaling factor (the max norm)
        """
        # Compute the L2 norm for each displacement vector (over the channel dimension)
        # Result shape: [B, H, W, D]
        if scale is  None:
            norms = displacement.norm(p=2, dim=1)
            
            # Find the maximum displacement magnitude (a scalar)
            scale = norms.max()
        
        # Normalize: scale every displacement vector by the same factor
        normalized = displacement / scale
        return normalized, scale

    def standardize_displacement_field(self, data, mean=None,std=None):
        """
        Standardize a 3D displacement field tensor isotropically across all channels.
        
        Args:
            data: Tensor of shape (B, C, D, H, W), e.g., (B, 3, 128, 128, 128)
            mean: Optional scalar with precomputed global mean
            std: Optional scalar with precomputed global std
            
        Returns:
            standardized_data: standardized tensor
            mean: Global mean (computed if not provided)
            std: Global std (computed if not provided)
        """
            # Compute global mean and std across all dimensions (B, C, D, H, W)
        if mean is None:
            mean = data.mean()
        if std is None:
            std = data.std()
        standardized_data = (data - mean) / (std)
        return standardized_data, mean, std

    def range_compression(self, sample, div_factor: float = 24, function: str = 'arcsinh', epsilon: float = 1e-8):
        """Applies compression on the input."""
        if function == 'arcsinh':
            return torch.arcsinh(sample / div_factor)*div_factor
        if function == 'tanh':
            return torch.tanh(sample / div_factor)*div_factor
        if function == 'sqrt':
            return torch.sign(sample)*torch.sqrt(torch.abs(sample + epsilon) / div_factor)*div_factor
        else:
            return sample  

    def reverse_range_compression(self, sample, div_factor: float = 24, function: str = 'arcsinh', epsilon: float = 1e-8):
        """Undos compression on the output."""
        if function == 'arcsinh':
            return torch.sinh(sample / div_factor)*div_factor
        if function == 'tanh':
            return torch.arctanh(torch.clamp((sample / div_factor),min=-0.999,max=0.999))*div_factor
        if function == 'sqrt':
            return torch.sign(sample)*torch.square((sample - epsilon) / div_factor)*div_factor  
        else:
            return sample

    def sample_initial_condition(self, input_data, diffusion_sampler='DDIM'):
        """Sample initial conditions using the reverse diffusion process."""
        if self.model_type == "ICdiffusion":
            # Placeholder: adapt based on your diffusion process (e.g., VESDE)
            shape = input_data.shape
            x = self.sde.prior_sampling(shape).to(self.device)  # Noise as starting point
            timesteps = self.sde.timesteps.to(self.device)  # Reverse diffusion steps
            for t in timesteps:
                t_vec = torch.ones(1, device=self.device) * t
                model_output = self.model(torch.cat([x, input_data], dim=1), t_vec)
                x, x_mean = self.sde.update_fn(x, t_vec, model_output=model_output)
            return x, x_mean
        elif self.model_type == "DDIM":
            if diffusion_sampler == 'DDIM':
                sampler = DDIMSampler(self.model, beta=self.hparams.ddim_beta, T=self.hparams.diffusion_num_scales)
            elif diffusion_sampler == 'DDPM':
                sampler = DDPMSampler(self.model, beta=self.hparams.ddim_beta, T=self.hparams.diffusion_num_scales)
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")
            z_t = torch.randn_like(input_data, device=input_data.device)
            x = sampler(z_t, input_data, only_return_x_0=True)
            return x
        else:
            raise ValueError(f"No sampling function implemented for model type: {self.model_type}")

