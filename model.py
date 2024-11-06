import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from periodic_padding import periodic_padding_3d

def crop_tensor(x):
	x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
	return x

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)
    
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


class Lpt2NbodyNetnoperiodicpadding(nn.Module):
    def __init__(self, block):
        super(Lpt2NbodyNetnoperiodicpadding, self).__init__()
        self.layer1 = self._make_layer(block, 3, 64, blocks=2, stride=1)
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
        self.deconv4 = nn.ConvTranspose3d(64, 3, 1, stride=1, padding=0)

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
        x = nn.functional.relu(self.deconv_batchnorm1(crop_tensor(self.deconv1(x))), inplace=True)
        x = torch.cat((x, x2), dim=1)
        x = self.layer6(x)
        x = nn.functional.relu(self.deconv_batchnorm2(crop_tensor(self.deconv2(x))), inplace=True)
        x = torch.cat((x, x1), dim=1)
        x = self.layer7(x)
        x = self.deconv4(x)
        return x

class Lpt2NbodyNet(nn.Module):
    def __init__(self, block):
        super(Lpt2NbodyNet, self).__init__()
        self.layer1 = self._make_layer(block, 3, 64, blocks=2, stride=1)
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
        self.deconv4 = nn.ConvTranspose3d(64, 3, 1, stride=1, padding=0)

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
    def __init__(self, block, num_layers=2, base_filters=64, blocks_per_layer=2):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = 3
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

        self.final_conv = nn.ConvTranspose3d(out_channels, 3, 1, stride=1, padding=0)

        # Predefine BatchNorm3d and ReLU layers for each decoder step
        self.batch_norms = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        for i in range(num_layers):
            self.batch_norms.insert(0, nn.BatchNorm3d(base_filters * (2 ** i)))  # Adjust channels accordingly

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
            x = self.batch_norms[i // 2](x)  # BatchNorm
            x = nn.ReLU(inplace=True)(x)  # ReLU
            
            # Skip connection with encoder outputs
            x = torch.cat((x, encoder_outputs[len(encoder_outputs)-2-i//2]), dim=1)  # Skip connection
            
            x = self.decoders[i + 1](x)  # Non-compression layer

        # Final 1x1 Conv
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
                lr_cosine_period: int = 24000,
                custom: bool = False,
                num_layers: int = 4, 
                base_filters: int = 64, 
                blocks_per_layer: int = 2, 
                periodic_padding: bool = True,
                **kwargs
                ):
        super(Lpt2NbodyNetLightning, self).__init__()

        self.save_hyperparameters(ignore=['kwargs'])  # This will save all init args except kwargs

        if not periodic_padding:
            self.model = Lpt2NbodyNetnoperiodicpadding(BasicBlock)
        elif not custom:
            self.model = Lpt2NbodyNet(BasicBlock)
        else:
            self.model = UNet3D(block=BasicBlock,num_layers=self.hparams.num_layers,base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer)
        self.criterion = nn.MSELoss()  

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.criterion(y_hat, y)
        
        # Log the batch loss separately
        self.log('train_batch_loss', train_loss, on_step=True, on_epoch=False, logger=True)
        
        # Log the epoch loss separately
        self.log('train_epoch_loss', train_loss, on_step=False, on_epoch=True, logger=True,sync_dist=True)

        optimizer = self.optimizers()

        # Access the learning rate from the optimizer's parameter groups
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False, logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        
        # Log the batch loss separately
        self.log('val_batch_loss', val_loss, on_step=True, on_epoch=False, logger=True)
        
        # Log the epoch loss separately
        self.log('val_epoch_loss', val_loss, on_step=False, on_epoch=True, logger=True,sync_dist=True)

        return y_hat
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        
        # Log the batch loss separately
        self.log('test_batch_loss', test_loss, on_step=True, on_epoch=False, logger=True)
        
        # Log the epoch loss separately
        self.log('test_epoch_loss', test_loss, on_step=False, on_epoch=True, logger=True,sync_dist=True)

        return y_hat
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        else:
            optimizer = optim.Adam(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_scheduler == 'Constant':
            return optimizer

        elif self.hparams.lr_scheduler == 'Cosine':
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.hparams.lr,
                    total_iters=self.hparams.lr_warmup,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.lr_cosine_period
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