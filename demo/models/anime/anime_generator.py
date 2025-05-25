import torch
import torch.nn as nn

class CNN_block_gen(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2, upscale=False, act="LeakyReLU", use_dropout=False):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,4,2,1, bias=False) if upscale  
            else nn.Conv2d(in_channels,out_channels,4,stride,1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2) 
         )
        self.use_dropout=use_dropout
        self.dropout=nn.Dropout(0.5)
        
    def forward(self,x):
        x = self.conv_layer(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self,in_channels=3, features=64):
        super().__init__()
        self.first=nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        
        self.down1 = CNN_block_gen(features, features*2, upscale=False, act="LeakyReLU", use_dropout=False)  
        self.down2 = CNN_block_gen(features*2, features*4, upscale=False, act="LeakyReLU", use_dropout=False)  
        self.down3 = CNN_block_gen(features*4, features*8, upscale=False, act="LeakyReLU", use_dropout=False)  
        self.down4 = CNN_block_gen(features*8, features*8, upscale=False, act="LeakyReLU", use_dropout=False)  
        self.down5 = CNN_block_gen(features*8, features*8, upscale=False, act="LeakyReLU", use_dropout=False)  
        self.down6 = CNN_block_gen(features*8, features*8, upscale=False, act="LeakyReLU", use_dropout=False)  
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"), nn.ReLU(),  
        )
        
        self.up1 = CNN_block_gen(features*8, features*8, upscale=True, act="relu", use_dropout=True)
        self.up2 = CNN_block_gen(features*8*2, features*8, upscale=True, act="relu", use_dropout=True)
        self.up3 = CNN_block_gen(features*8*2, features*8, upscale=True, act="relu", use_dropout=False)
        self.up4 = CNN_block_gen(features*8*2, features*8, upscale=True, act="relu", use_dropout=False)
        self.up5 = CNN_block_gen(features*8*2, features*4, upscale=True, act="relu", use_dropout=False)
        self.up6 = CNN_block_gen(features*4*2, features*2, upscale=True, act="relu", use_dropout=False)
        self.up7 = CNN_block_gen(features*2*2, features, upscale=True, act="relu", use_dropout=False)
        self.last=nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels,kernel_size=4,stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.first(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
    
        bottleneck = self.bottleneck(d7)
    
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
    
        return self.last(torch.cat([up7, d1], 1))