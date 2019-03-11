import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools

class _ConvLayer(nn.Sequential):
    def __init__(self, norm_layer,input_features, output_features):
        super(_ConvLayer, self).__init__()
        self.add_module('conv_1', Conv2d(input_features, output_features,
                                        kernel_size=3, stride=2, padding =1))
        self.add_module('conv_2', Conv2d(output_features, output_features,
                                kernel_size=3, stride=1, padding =1))
        self.add_module('norm',norm_layer(output_features))
        self.add_module('relu', nn.ReLU(True))

class _UpScale(nn.Sequential):
    def __init__(self, norm_layer,input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module('convtrans',nn.ConvTranspose2d(input_features, output_features, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.add_module('conv', Conv2d(output_features, output_features,
                                kernel_size=3, stride=1, padding =1))
        self.add_module('norm',norm_layer(output_features))
        self.add_module('relu',nn.ReLU(True))


        self.add_module('conv2_', Conv2d(input_features, output_features * 4,
                                         kernel_size=3))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
        self.add_module('pixelshuffler', _PixelShuffler())

class _Flatten(nn.Module):
    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output

class _Reshape(nn.Module):
    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)  # channel * 4 * 4
        return output

class AutoEncoder(nn.Module):
    def __init__(self,in_batch):
        super(AutoEncoder,self).__init__()
        if in_batch:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=False)
        else:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, 32, kernel_size=7, padding=0),
                norm_layer(32),
                nn.ReLU(True),
                _ConvLayer(norm_layer,32,64), #128->64
                _ConvLayer(norm_layer,64,128), #64->32
                _ConvLayer(norm_layer,128,256), #32->16
                _ConvLayer(norm_layer,256,512), #16->8
                _ConvLayer(norm_layer,512,1024), #8->4
                _Flatten(),
                nn.Linear(1024*4*4,1024),
                nn.Linear(1024,1024*4*4),
                Reshape(),
                _UpScale(norm_layer,1024,512), #4->8
            )
        self.decoder_real = nn.Sequential(
            _UpScale(norm_layer,512,256), #8->16
            _UpScale(norm_layer,256,128), #16->32
            _UpScale(norm_layer,128,64), #32->64
            _UpScale(norm_layer,64,32), #64->128
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, kernel_size=7, padding=0),
            nn.Tanh()
            )

        self.decoder_anime = nn.Sequential(
            _UpScale(norm_layer,512,256), #8->16
            _UpScale(norm_layer,256,128), #16->32
            _UpScale(norm_layer,128,64), #32->64
            _UpScale(norm_layer,64,32), #64->128
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, kernel_size=7, padding=0),
            nn.Tanh()
            )

    def forward(self, input, select):
        if select == 'real':
            out = self.encoder(input)
            out = self.decoder_real(out)
        else:
            out = self.encoder(input)
            out = self.decoder_anime(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_nc, out_nc, ndf=64, n_layers=3, use_sigmoid=False, in_batch=False):
        super(Discriminator, self).__init__()
        #         if type(norm_layer) == functools.partial:
        #             use_bias = norm_layer.func == nn.InstanceNorm2d
        #         else:
        #             use_bias = norm_layer == nn.InstanceNorm2d
        use_bias = True
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        if in_batch:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=False)
        else:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        kw = 4
        padw = 1

        sequence=[]
#         if num_pooling>0:
#             for i in range(num_pooling):
#                 sequence.append(nn.AvgPool2d(2,stride=2))
        sequence += [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, out_nc, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

        self.model = init_weights(self.model)
        self.pooling = nn.AvgPool2d(2,stride=2)

    def forward(self, input, num_pooling=0):
        result=input
        for p in range(num_pooling):
            result = self.pooling(input)
        return self.model(result)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=128, n_blocks=6, n_downsampling=2, padding_type='reflect', in_batch=False):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        use_bias = True
        norm_layer = None
        if in_batch:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=False)
        else:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
            
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
                                stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [resnet(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3,
                                stride=1, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        self.rgb_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )
        self.model = init_weights(self.model)
        self.rgb_layer = init_weights(self.rgb_layer)
        self.pooling = nn.AvgPool2d(2,stride=2)
    def to_rgb(self,input):
        return self.rgb_layer(input)
    def forward(self, input, no_rgb = True, num_pooling=0):
        for p in range(num_pooling):
            input = self.pooling(input)
        if no_rgb:
            return self.model(input)
        else:
            return self.rgb_layer(self.model(input))

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
    net.apply(init_func)
    return net

class resnet(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias):
        super(resnet, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
