import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from einops import rearrange, parse_shape

##################################################################################################################
### For the base source, see: https://arxiv.org/abs/2101.04775, https://github.com/odegeasslbc/FastGAN-pytorch ###
##################################################################################################################

def NearestUpSampling(c_in, c_out):
    upsampling = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        spectral_norm(nn.Conv2d(c_in, c_out*2, 3, 1, 1, bias=False)),
        nn.BatchNorm2d(c_out*2),
        GLU(),
       )
    return upsampling

class Reduction(nn.Module):
    def __init__(self, c_in, c_out):
        super(Reduction, self).__init__()
        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            spectral_norm(nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_conv = nn.Sequential(spectral_norm(nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False)),
                                        nn.BatchNorm2d(c_out),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        spectral_norm(nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)),
                                        nn.BatchNorm2d(c_out),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        )

    def forward(self, x):
        return (self.avg_pool(x) + self.down_conv(x)) / 2

class Simple_Decoder(nn.Module):
    def __init__(self, c_in, c_out):
        super(Simple_Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)

class Simple_Decoder_8(nn.Module):
    def __init__(self):
        super(Simple_Decoder_8, self).__init__()
        self.main = nn.Sequential(
            Simple_Decoder(256, 128), # 128 x 16 x 16
            Simple_Decoder(128, 64), # 64 x 32 x 32
            Simple_Decoder(64, 32), # 32 x 64 x 64
            Simple_Decoder(32, 16), # 16 x 128 x 128
            spectral_norm(nn.Conv2d(16, 3, 3, 1, 1, bias=False)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Simple_Decoder_16(nn.Module):
    def __init__(self):
        super(Simple_Decoder_16, self).__init__()
        self.main = nn.Sequential(
            Simple_Decoder(128, 64), # 64 x 32 x 32
            Simple_Decoder(64, 32), # 32 x 64 x 64
            Simple_Decoder(32, 16), # 16 x 128 x 128
            spectral_norm(nn.Conv2d(16, 3, 3, 1, 1, bias=False)), # 3 x 128 x 128
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.main(x)

class Conv_SelfAttn(nn.Module):
    # Convolutional Self-Attention, see https://arxiv.org/abs/1805.08318
    def __init__(self, c_in):
        super(Conv_SelfAttn, self).__init__()
        self.queries = nn.Conv2d(c_in, c_in//8, kernel_size = 1, bias = False)
        self.keys = nn.Conv2d(c_in, c_in//8, kernel_size = 1, bias = False)
        self.values = nn.Conv2d(c_in, c_in, kernel_size = 1, bias = False)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable parameter
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        queries = rearrange(self.queries(x), 'b c h w -> b (h w) c')
        keys = rearrange(self.keys(x), 'b c h w -> b c (h w)')
        values = rearrange(self.values(x), 'b c h w -> b (h w) c')
        energy = torch.bmm(queries, keys) 
        attention = self.softmax(energy)
        out = torch.bmm(attention, values)
        out = x + self.gamma * rearrange(out, 'b (h w) c -> b c h w', **parse_shape(x, 'b c h w'))
        return out


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False)), nn.LeakyReLU(0.1, inplace=True),
                                    spectral_norm(nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False)), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=256, nc=3):
        super(Generator, self).__init__()

        layers_filters = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in layers_filters.items():
            nfc[k] = int(v*ngf)
        
        self.initial = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(nz, nfc[4]*2, 4, 1, 0, bias=False)), 
                                    nn.BatchNorm2d(nfc[4]*2),
                                    GLU(), # Output 1024 x 4 x 4
        )

        self.up1 = NearestUpSampling(c_in=nfc[4], c_out=nfc[8]) # ----> 512 x 8 x 8
        self.up2 = NearestUpSampling(c_in=nfc[8], c_out=nfc[16]) # ----> 256 x 16 x 16
        self.up3 = NearestUpSampling(c_in=nfc[16], c_out=nfc[32]) # ----> 128 x 32 x 32
        self.up4 = NearestUpSampling(c_in=nfc[32], c_out=nfc[64]) # ----> 128 x 64 x 64
        self.up5 = NearestUpSampling(c_in=nfc[64], c_out=nfc[128]) # ----> 64 x 128 x 128
        self.up6 = NearestUpSampling(c_in=nfc[128], c_out=nfc[256]) # ----> 32 x 256 x 256
        self.up7 = NearestUpSampling(c_in=nfc[256], c_out=nfc[512]) # ----> 16 x 512 x 512

        self.final = nn.Sequential(spectral_norm(nn.ConvTranspose2d(nfc[512], nfc[1024]*2, 3, 1, 1, bias=False)),
                                    nn.BatchNorm2d(nfc[1024]*2),
                                    GLU(),
                                    spectral_norm(nn.Conv2d(nfc[1024], nc, 3, 1, 1, bias=False)),
                                    nn.Tanh()) # Output 3 x 1024 x 1024
        
        self.skip_layer_4_64 = SEBlock(nfc[4], nfc[64])
        self.skip_layer_8_128 = SEBlock(nfc[8], nfc[128])
        self.skip_layer_16_256 = SEBlock(nfc[16], nfc[256])
        self.skip_layer_32_512 = SEBlock(nfc[32], nfc[512])

        self.attn_64 = Conv_SelfAttn(nfc[64])

    def forward(self, z):
        z = self.initial(z) # Output 1024 x 4 x 4
        
        out_up1 = self.up1(z) # Output 512 x 8 x 8
        
        out_up2 = self.up2(out_up1) # Output 256 x 16 x 16
        
        out_up3 = self.up3(out_up2) # Output 128 x 32 x 32
        
        out_up4 = self.up4(out_up3) # Output 128 x 64 x 64
        out_up4 = self.attn_64(out_up4)
        out_up4 = self.skip_layer_4_64(z, out_up4)
        
        out_up5 = self.up5(out_up4) # Output 64 x 128 x 128
        out_up5 = self.skip_layer_8_128(out_up1, out_up5)
        
        out_up6 = self.up6(out_up5) # Output 32 x 256 x 256
        out_up6 = self.skip_layer_16_256(out_up2, out_up6)
        
        out_up7 = self.up7(out_up6) # Output 16 x 512 x 512
        out_up7 = self.skip_layer_32_512(out_up3, out_up7)
        
        out = self.final(out_up7) # Output 3 x 1024 x 1024
        
        return out

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()

        layers_filters = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        
        nfc = {}
        for k, v in layers_filters.items():
            nfc[k] = int(v*ndf)
        
        self.initial = nn.Sequential(spectral_norm(nn.Conv2d(nc, nfc[1024], 4, 2, 1, bias=False)), 
                                    nn.BatchNorm2d(nfc[1024]), # 8 x 256 x 256
                                    nn.LeakyReLU(0.2, inplace=True),
                                    
        )
        self.down1 = Reduction(nfc[1024], nfc[512]) # 16 x 128 x 128
        self.down2 = Reduction(nfc[512], nfc[256]) # 32 x 64 x 64
        self.down3 = Reduction(nfc[256], nfc[128]) # 64 x 32 x 32
        self.down4 = Reduction(nfc[128], nfc[64]) # 128 x 16 x 16
        self.down5 = Reduction(nfc[64], nfc[32]) # 256 x 8 x 8

        self.final = nn.Sequential(spectral_norm(nn.Conv2d(nfc[32], nfc[16], 1, 1, 0, bias=False)),
                                    nn.BatchNorm2d(nfc[16]),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    spectral_norm(nn.Conv2d(nfc[16], 1, 4, 1, 0, bias=False)), # 1 x 5 x 5

        )

        self.Simple_8 = Simple_Decoder_8()
        self.Simple_16 = Simple_Decoder_16()

        self.attn_64 = Conv_SelfAttn(nfc[256])

    def forward(self, x, tag):
        x = self.initial(x) #--> 8 x 256 x 256
        x = self.down1(x) #--> 16 x 128 x 128
        x = self.down2(x) #--> 32 x 64 x 64
        x = self.attn_64(x) #--> 32 x 64 x 64
        x = self.down3(x) #--> 64 x 32 x 32

        feat_16 = self.down4(x) #--> 128 x 16 x 16
        feat_8 = self.down5(feat_16) #--> 256 x 8 x 8
        out = rearrange(self.final(feat_8), 'b c h w -> b (c h w)') #--> b_size x 25

        if tag == "Real":
            x_recon_random = self.Simple_16(feat_16) # 3 x 128 x 128 Random crop
           
            x_recon = self.Simple_8(feat_8) # 3 x 128 x 128 Reconstruction
            
            return  out, [x_recon_random, x_recon]

        return out

# Testing the networks
if __name__ == '__main__':
    # Testing the generator
    generator = Generator()
    z = torch.randn(1, 256, 1, 1)
    out = generator(z)
    print(out.shape)

    # Testing the discriminator
    discriminator = Discriminator()
    x = torch.randn(4, 3, 512, 512)
    out = discriminator(x, "Fake")
    print(out.shape)
    print(rearrange(torch.randn(16, 1, 5, 5), 'b c h w -> b (c h w)').shape)