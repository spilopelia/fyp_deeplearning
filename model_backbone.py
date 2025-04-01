import torch.nn as nn
import torch
from periodic_padding import periodic_padding_3d

def crop_tensor(x):
	x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
	return x

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

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