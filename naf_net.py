import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, D, H, W)
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = self.weight * x + self.bias
        return x

class PixelShuffle3D(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        channels //= self.scale_factor ** 3
        x = x.view(batch_size, channels, self.scale_factor, self.scale_factor, self.scale_factor, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels, 
                   depth * self.scale_factor, 
                   height * self.scale_factor, 
                   width * self.scale_factor)
        return x

class BaselineBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(c, dw_channel, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv3d(dw_channel, dw_channel, kernel_size=3, padding=1, 
                              groups=dw_channel, bias=True)
        self.conv3 = nn.Conv3d(dw_channel, c, kernel_size=1, padding=0, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dw_channel, dw_channel // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dw_channel // 2, dw_channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv3d(ffn_channel, c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet3D_base(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, 
                 enc_blk_nums=[], dec_blk_nums=[], dw_expand=1, ffn_expand=2):
        super().__init__()
        self.intro = nn.Conv3d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv3d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]))
            self.downs.append(nn.Conv3d(chan, 2*chan, kernel_size=2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv3d(chan, chan * 4, kernel_size=1, bias=False),
                PixelShuffle3D(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, D, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :D, :H, :W]

    def check_image_size(self, x):
        _, _, d, h, w = x.size()
        mod_pad_d = (self.padder_size - d % self.padder_size) % self.padder_size
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d))

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock3D(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(c, dw_channel, 1)
        self.conv2 = nn.Conv3d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv3d(dw_channel//2, c, 1)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dw_channel//2, dw_channel//2, 1)
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(c, ffn_channel, 1)
        self.conv5 = nn.Conv3d(ffn_channel//2, c, 1)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma

class NAFNet3D(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        
        self.intro = nn.Conv3d(img_channel, width, kernel_size=3, padding=1)
        self.ending = nn.Conv3d(width, img_channel, kernel_size=3, padding=1)
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Encoder path
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock3D(chan) for _ in range(num)]))
            self.downs.append(nn.Conv3d(chan, chan*2, kernel_size=2, stride=2))
            chan *= 2

        # Middle blocks
        self.middle_blks = nn.Sequential(*[NAFBlock3D(chan) for _ in range(middle_blk_num)])

        # Decoder path
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv3d(chan, chan*4, 1),
                PixelShuffle3D(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock3D(chan) for _ in range(num)]))
            
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.check_image_size(x)
        
        x = self.intro(x)
        enc_features = []
        
        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_features.append(x)
            x = down(x)
            
        # Middle
        x = self.middle_blks(x)
        
        # Decoder
        for decoder, up, enc_ft in zip(self.decoders, self.ups, reversed(enc_features)):
            x = up(x)
            x = x + enc_ft
            x = decoder(x)
            
        x = self.ending(x)
        return x[:, :, :D, :H, :W]

    def check_image_size(self, x):
        _, _, D, H, W = x.size()
        pad_d = (self.padder_size - D % self.padder_size) % self.padder_size
        pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))