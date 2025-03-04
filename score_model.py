import score_layers, score_layerspp
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockBigGAN = score_layerspp.ResnetBlockBigGANpp
conv3x3 = score_layerspp.conv3x3
conv1x1 = score_layerspp.conv1x1
get_act = score_layers.get_act

default_initializer = score_layers.default_init



class UNet3DModel(nn.Module):

  def __init__(
        self,
        act_f='swish',  # From "model.nonlinearity": "swish"
        nf=32,  # From "model.nf": 32
        ch_mult=[1, 2, 2, 1, 1],  # From "model.ch_mult": [1, 2, 2, 1, 1]
        num_res_blocks=2,  # From "model.num_res_blocks": 2
        dropout=0.1,  # From "model.dropout": 0.1
        resamp_with_conv=True,  # From "model.resamp_with_conv": true
        image_size=128,  # From "data.image_size": 128
        conditional=True,  # From "model.conditional": true
        fir=False,  # From "model.fir": false
        fir_kernel=[1, 3, 3, 1],  # From "model.fir_kernel": [1, 3, 3, 1]
        skip_rescale=True,  # From "model.skip_rescale": true
        embedding_type="fourier",  # From "model.embedding_type": "fourier"
        init_scale=0.0,  # From "model.init_scale": 0.0
        num_input_channels=6,  # From "data.num_input_channels": 2
        num_output_channels=3,  # From "data.num_output_channels": 1
        fourier_scale=16,  # From "model.fourier_scale": 16
        **kwargs,
    ):
    super().__init__()
    self.act = act = get_act(act_f)
    self.nf = nf
    self.num_res_blocks = num_res_blocks
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional  # noise-conditional
    self.skip_rescale = skip_rescale
    self.embedding_type = embedding_type 
    assert embedding_type in ['fourier']


    modules = []
    # timestep/noise_level embedding; only for continuous training

    modules.append(score_layerspp.GaussianFourierProjection(
      embedding_size=nf, scale=fourier_scale
    ))
    embed_dim = 2 * nf


    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)


    ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                    act=act,
                                    dropout=dropout,
                                    fir=fir,
                                    fir_kernel=fir_kernel,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale,
                                    temb_dim=nf * 4)


    # Downsampling block

    input_channels = num_input_channels
    output_channels = num_output_channels


    # Downsampling block

    modules.append(conv3x3(input_channels, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        modules.append(ResnetBlock(down=True, in_ch=in_ch))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if i_level != 0:
        modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch, eps=1e-6))
    modules.append(conv3x3(in_ch, output_channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)


  def forward(self, x, time_cond):
    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None


    # Downsampling block

    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        hs.append(h)

      if i_level != self.num_resolutions - 1:
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1


    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1

      if i_level != 0:
        h = modules[m_idx](h, temb)
        m_idx += 1

    assert not hs

    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1

    assert m_idx == len(modules)

    return h