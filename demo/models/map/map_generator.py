import torch.nn as nn
import torch 

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels ,kernel_size = 4, stride = 2,padding =1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,X):
        return self.block(X)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,dropout = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels , kernel_size = 4, stride = 2, padding =1 ,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self,X):
        X = self.block(X)
        return self.dropout_layer(X) if self.dropout else X

class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.de1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4,2,1, padding_mode= "reflect"),
            nn.ReLU(),
        )                                                  #128,128
        self.de2 = DownBlock(64,128)                       #64,64
        self.de3 = DownBlock(128, 256)                     #32,32
        self.de4 = DownBlock(256,512)                      #16,16
        self.de5 = DownBlock(512 ,512)                     # 8,8
        self.de6 = DownBlock(512 ,512)                     # 4,4
        self.de7 = DownBlock(512 ,512)                     # 2,2
        self.de8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512 ,kernel_size = 4, stride = 2,padding =1),
            nn.LeakyReLU(0.2)
        )                                                    # 1,1

        self.up1 = UpBlock(512,512,True)                     # 2,2
        self.up2 = UpBlock(512*2,512,True)                   # 4,4
        self.up3 = UpBlock(512*2,512,True)                   # 8,8
        self.up4 = UpBlock(512*2,512,True)                   # 16,16
        self.up5 = UpBlock(512*2,256)                        # 32,32
        self.up6 = UpBlock(256*2,128)                        # 64,64
        self.up7 = UpBlock(128*2,64)                         # 128,128
        self.up8 = nn.Sequential(                          # 256,256
            nn.ConvTranspose2d(in_channels = 64*2, out_channels = 3, kernel_size = 4 , stride = 2 , padding = 1),  
            nn.Tanh()
        )
    def forward(self, x):
        d1 = self.de1(x)
        d2 = self.de2(d1)
        d3 = self.de3(d2)
        d4 = self.de4(d3)
        d5 = self.de5(d4)
        d6 = self.de6(d5)
        d7 = self.de7(d6)
        d8 = self.de8(d7)
 
        
        up1 = self.up1(d8)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))

        return self.up8(torch.cat([up7, d1], 1))