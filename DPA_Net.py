import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tqdm import tqdm



class TDCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TDCB, self).__init__()

        # Define the structure for each parallel layer
        self.layer1 = self._create_layer(in_channels, out_channels, dilation_rate=1, padding=1)
        self.layer2 = self._create_layer(in_channels, out_channels, dilation_rate=3, padding=3)
        self.layer3 = self._create_layer(in_channels, out_channels, dilation_rate=5, padding=5)
        #self.layer4 = self._create_layer(in_channels, out_channels, dilation_rate=7, padding=7)

    def _create_layer(self, in_channels, out_channels, dilation_rate, padding):
        # Use LayerNorm instead of BatchNorm2d
        layer_norm = LayerNorm(normalized_shape=out_channels)

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # Forward pass through each parallel layer
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)


        # Concatenate the outputs along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)


        return out

class FCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCB, self).__init__()

        self.Layerconv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.BacthNorm=nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.Layerconv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        x = self.Layerconv3x3(x)
        x=self.BacthNorm(x)
        x = self.gelu(x)
        x1 = self.Layerconv1x1(x)
        x2 = self.sigmoid(x1)

        outputs = x2*x
        return outputs



class MFEB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MFEB, self).__init__()

        self.conv_1x1_PA = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,stride=stride, padding=0)
        self.sigmoid1=nn.Sigmoid()


        self.conv_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels//3)
        self.gelu1 = nn.GELU()

        self.conv_5x5 = nn.Conv2d(in_channels=in_channels, out_channels = in_channels//3 , kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(in_channels//3)
        self.gelu2 = nn.GELU()

        self.Pooling= nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.sizeMatching= nn.Conv2d(in_channels, in_channels//3, kernel_size=1)




    def forward(self, x):
        inputFeature = x

        conv1x1_out=self.conv_1x1_PA(x)
        sigmoidOut= self.sigmoid1(conv1x1_out)

        PA= sigmoidOut*inputFeature

        x1= self.conv_3x3(x)
        x1= self.bn1(x1)
        x1= self.gelu1(x1)


        x2=self.conv_5x5(x)
        x2=self.bn2(x2)
        x2=self.gelu2(x2)


        x3=self.Pooling(x)
        x3=self.sizeMatching(x3)

        outputs = [x1,x2,x3]
        out1=torch.cat(outputs, 1)

        out= out1+PA



        return out

class PAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PAB, self).__init__()

        self.Layerconv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Forward pass through the block
        inputFeature = x
        x1 = self.Layerconv1x1(x)
        x2 = self.sigmoid(x1)
        outputs= x2 * inputFeature
        return outputs

class Block(nn.Module):


    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):


    def __init__(self, in_chans=3, num_classes=38,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            if i == 0:
               stage.add_module('Triple dilated convolution block', TDCB(in_channels=96, out_channels=32))

            elif i==1:
                stage.add_module('Fused convolution block',FCB(in_channels=192, out_channels=192))

            elif i==2:
               stage.add_module('Multiscale feature extraction block',MFEB(in_channels=384,out_channels=384))

            elif i==3:
               stage.add_module('Pixel attention block',PAB(in_channels=768,out_channels=768))



            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):


    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





DPA_Net = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
print(DPA_Net)







