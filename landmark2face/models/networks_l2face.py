import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class AudioNet(nn.Module):

    def __init__(self):
        super(AudioNet, self).__init__()
        # audio
        self.audio1 = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
        )
        self.audio2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=(4, 1)), nn.ReLU()
        )

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, audio):
        x_a = self.audio1(audio)
        x_a = self.audio2(x_a)
        x_a = x_a.view(-1, self.num_flat_features(x_a))
        return x_a


class ResBlockDecoderWithAudioFeature(nn.Module):
    def __init__(self, shape, input_nc, output_nc, hidden_nc):
        super(ResBlockDecoderWithAudioFeature, self).__init__()
        self.shape = shape
        self.main_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bypass_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.main_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1)
        self.bypass_conv1 = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, stride=1, padding=1)
        self.bypass_conv2 = nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(512,shape[0]*shape[1]*shape[2])

        self.norm1 = nn.InstanceNorm2d(hidden_nc)
        self.norm2 = nn.InstanceNorm2d(hidden_nc)
        self.norm3 = nn.InstanceNorm2d(input_nc)

    def forward(self, x, feature):
        bypass_out = self.linear(feature)
        bypass_out = bypass_out.view(-1,self.shape[0],self.shape[1],self.shape[2])
        bypass_out = torch.cat((x,bypass_out),1)
        bypass_out = self.norm1(bypass_out)
        bypass_out = F.leaky_relu(bypass_out,0.1)
        bypass_out = self.bypass_up(bypass_out)
        bypass_out = self.bypass_conv1(bypass_out)
        bypass_out = self.norm2(bypass_out)
        bypass_out = F.leaky_relu(bypass_out,0.1)
        bypass_out = self.bypass_conv2(bypass_out)

        x = self.norm3(x)
        x = F.leaky_relu(x,0.1)
        main_out = self.main_up(x)
        main_out = self.main_conv(main_out)

        out = main_out + bypass_out
        return out

class Audio_model(nn.Module):
    def __init__(self):
        super(Audio_model,self).__init__()
        self.audio = AudioNet()  #[B,512]
        self.linear = nn.Linear(512,1024)
        self.res1 = ResBlockDecoderWithAudioFeature((16,8,8),16,32,16+16)
        self.res2 = ResBlockDecoderWithAudioFeature((16,16,16),32,64,32+16)
        self.res3 = ResBlockDecoderWithAudioFeature((16,32,32),64,128,64+16)
    def forward(self,mfcc):
        audio_out = self.audio(mfcc)
        x_out = self.linear(audio_out)
        x_out = x_out.view(-1,16,8,8)
        x_out = self.res1(x_out,audio_out)
        x_out = self.res2(x_out,audio_out)
        x_out = self.res3(x_out,audio_out)
        return x_out



class ResnetL2FaceGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        super(ResnetL2FaceGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_ref = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc*2, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_ref += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        model_prev = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc*2, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_prev += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        model2 = []
        for i in range(self.n_blocks):
            if i==0:
                model2 += [ResnetBlock(640,320, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            elif i==1:
                model2 += [ResnetBlock(320,256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]

        self.model_audio = Audio_model()
        self.model_ref = nn.Sequential(*model_ref)
        self.model_prev = nn.Sequential(*model_prev)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, ref_img,prev_img,landmark,mfcc):
        """Standard forward"""
        x_ref = self.model_ref(torch.cat([ref_img,landmark],1))
        x_prev = self.model_prev(torch.cat([prev_img,landmark],1))
        x_audio = self.model_audio(mfcc)
        x = self.model2(torch.cat([x_ref,x_prev,x_audio],1))
        out = self.model3(x)

        return out


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias)
        self.shortcut = nn.Sequential(*[nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim_out)])

    def build_conv_block(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.shortcut(x) + self.conv_block(x)  # add skip connections
        return out
