import torch
print(torch.__version__)
print(torch.version.cuda)


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tqdm import tqdm



class DCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DCBlock, self).__init__()

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
        #out4 = self.layer4(x)

        # Concatenate the outputs along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)


        return out

class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomBlock, self).__init__()

        self.Layerconv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.BacthNorm=nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.Layerconv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Forward pass through the block
        originalFeature = x
        x = self.Layerconv3x3(x)
        x=self.BacthNorm(x)
        x = self.gelu(x)
        x1 = self.Layerconv1x1(x)
        x2 = self.sigmoid(x1)

        outputs = x2*x
        return outputs



class ParalleConv_PA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ParalleConv_PA, self).__init__()

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

class PixelAttentoin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelAttentoin, self).__init__()

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
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

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
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

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
               stage.add_module('Dilation Block 1', DCBlock(in_channels=96, out_channels=32))

            elif i==1:
                stage.add_module('ConvBlockPixelAttention Block2',CustomBlock(in_channels=192, out_channels=192))

            elif i==2:
               stage.add_module('Parallel Convolution Block',ParalleConv_PA(in_channels=384,out_channels=384))

            elif i==3:
               stage.add_module('PixelAttention',PixelAttentoin(in_channels=768,out_channels=768))



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
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}




@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model



custom_model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
custom_model.head = nn.Linear(768, 38)
print(custom_model)


from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Allow Cross-Origin requests

# Set the uploads folder path relative to the project directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path='C:\\Users\\aliha\\Downloads\\DPA-Net_Proposed_ChekAgain_PV_LL_N_STD_45Fold1_28_Aug_2024_Last23.pt'### path of model on server
device = torch.device('cpu')
custom_model.load_state_dict(torch.load(model_path, map_location=device))
custom_model = custom_model.to(device)
custom_model.eval()


# Transform for your model (customize as needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the image and preprocess it
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Run prediction
            with torch.no_grad():
                outputs = custom_model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class_idx = predicted.item()
            if predicted_class_idx == 0:
                predicted_class_name = "Apple Cedar Rust"  # Class name for index 0
            elif predicted_class_idx == 1:
                predicted_class_name = "Apple healthy"  # Class name for index 1
            elif predicted_class_idx == 2:
                predicted_class_name = "Apple black rot"  # Class name for index 2
            elif predicted_class_idx == 3:
                predicted_class_name = "Apple scab"  # Class name for index 3
            elif predicted_class_idx == 4:
                predicted_class_name = "Blueberry healthy"  # Class name for index 4
            elif predicted_class_idx == 5:
                predicted_class_name = "Cherry healthy"  # Class name for index 5
            elif predicted_class_idx == 6:
                predicted_class_name = "Cherry powdery mildew"  # Class name for index 6
            elif predicted_class_idx == 7:
                predicted_class_name = "Corn leaf blight"  # Class name for index 7
            elif predicted_class_idx == 8:
                predicted_class_name = "Corn gray leaf spot"  # Class name for index 8
            elif predicted_class_idx == 9:
                predicted_class_name = "Corn healthy"  # Class name for index 9
            elif predicted_class_idx == 10:
                predicted_class_name = "Corn common rust"  # Class name for index 10
            elif predicted_class_idx == 11:
                predicted_class_name = "Grape leaf blight"  # Class name for index 11
            elif predicted_class_idx == 12:
                predicted_class_name = "Grape black measles"  # Class name for index 12
            elif predicted_class_idx == 13:
                predicted_class_name = "Grape healthy"  # Class name for index 13
            elif predicted_class_idx == 14:
                predicted_class_name = "Grape black rot"  # Class name for index 14
            elif predicted_class_idx == 15:
                predicted_class_name = "Orange healthy"  # Class name for index 15
            elif predicted_class_idx == 16:
                predicted_class_name = "Pepper bacterial spot"  # Class name for index 16
            elif predicted_class_idx == 17:
                predicted_class_name = "Pepper healthy"  # Class name for index 17
            elif predicted_class_idx == 18:
                predicted_class_name = "Peach bacterial spot"  # Class name for index 18
            elif predicted_class_idx == 19:
                predicted_class_name = "Peach healthy"  # Class name for index 19
            elif predicted_class_idx == 20:
                predicted_class_name = "Potato early blight"  # Class name for index 20
            elif predicted_class_idx == 21:
                predicted_class_name = "Potato healthy"  # Class name for index 21
            elif predicted_class_idx == 22:
                predicted_class_name = "Potato late blight"  # Class name for index 22
            elif predicted_class_idx == 23:
                predicted_class_name = "Raspberry healthy"  # Class name for index 23
            elif predicted_class_idx == 24:
                predicted_class_name = "Soybean healthy"  # Class name for index 24
            elif predicted_class_idx == 25:
                predicted_class_name = "Squash powdery mildew"  # Class name for index 25
            elif predicted_class_idx == 26:
                predicted_class_name = "Strawberry healthy"  # Class name for index 26
            elif predicted_class_idx == 27:
                predicted_class_name = "Strawberry leaf scorch"  # Class name for index 27
            elif predicted_class_idx == 28:
                predicted_class_name = "Tomato bacterial spot"  # Class name for index 28
            elif predicted_class_idx == 29:
                predicted_class_name = "Tomato early blight"  # Class name for index 29
            elif predicted_class_idx == 30:
                predicted_class_name = "Tomato healthy"  # Class name for index 30
            elif predicted_class_idx == 31:
                predicted_class_name = "Tomato late blight"  # Class name for index 31
            elif predicted_class_idx == 32:
                predicted_class_name = "Tomato leaf mold"  # Class name for index 32
            elif predicted_class_idx == 33:
                predicted_class_name = "Tomato septoria leaf spot"  # Class name for index 33
            elif predicted_class_idx == 34:
                predicted_class_name = "Tomato spider mites"  # Class name for index 34
            elif predicted_class_idx == 35:
                predicted_class_name = "Tomato target spot"  # Class name for index 35
            elif predicted_class_idx == 36:
                predicted_class_name = "Tomato mosaic virus"  # Class name for index 36
            elif predicted_class_idx == 37:
                predicted_class_name = "Tomato yellow leaf curl"  # Class name for index 37
            else:
                predicted_class_name = "Unknown Class"  # Fallback for unexpected index

            #predicted_class_name = class_labels[predicted_class_idx]

            # Define custom message based on class index
            custom_messages = [
                "Miravis",
                "No need of pesticide",
                "Actigard 50WG",
                "Captan 80 WDG and  Li700",
                "No need of pesticide",
                "No need of pesticide",
                "Luna sensation",
                "PRCEPX2",
                "AZXFLT and PRCEPX2",
                "No need of pesticide",
                "AZXFLT and DFN",
                "Inspire Super",
                "Not found any pesticide",
                "No need of pesticide",
                "Immunox",
                "No need of pesticide",
                "Streptomycin P",
                "No need of pesticide",
                "Kocide 3000",
                "No need of pesticide",
                "Lycomax",
                "No need of pesticide",
                "Allcop 50% WP",
                "No need of pesticide",
                "No need of pesticide",
                "HMO 736",
                "No need of pesticide",
                "Kocide 3000",
                "Tetracycline and Streptomycin",
                "Iprodione 50% WP",
                "No need of pesticide",
                "Zineb 75% WP",
                "Benjovindiflutpr Difenoconazle",
                "Mancozeb 75% WP",
                "Spiromesifen 22.9% SC",
                "Mancozeb 75% WP",
                "Remove and burn all affected leaves",
                "Imidacloprid 17.8 SL"
            ]
            custom_message = custom_messages[predicted_class_idx]

            combined_message = f"Predicted Class: {predicted_class_name}<br>Pesticide info :  {custom_message}"

            return render_template('index.html', uploaded_image=filename, message=combined_message)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)