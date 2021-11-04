import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util.pytorch import to_tensor
from collections import OrderedDict
from rl.policies.distributions import (
    FixedNormal,
)

class CNN(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.activation_fn = nn.ReLU()

        self.convs = nn.ModuleList()
        w = config.img_width
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        for k, s, d in zip(config.kernel_size, config.stride, config.conv_dim):
            self.convs.append(init_(nn.Conv2d(input_dim, d, int(k), int(s))))
            w = int(np.floor((w - (int(k) - 1) - 1) / int(s) + 1))
            input_dim = d

        # screen_width == 32 (8,4)-(3,2) -> 3x3
        # screen_width == 64 (8,4)-(3,2)-(3,2) -> 3x3
        # screen_width == 128 (8,4)-(3,2)-(3,2)-(3,2) -> 3x3
        # screen_width == 256 (8,4)-(3,2)-(3,2)-(3,2) -> 7x7

        print("Output of CNN = %d x %d x %d" % (w, w, d))
        self.w = w
        self.output_size = w * w * d

    def forward(self, ob):
        out = ob
        for conv in self.convs:
            out = self.activation_fn(conv(out))
        out = out.flatten(start_dim=1)
        return out


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MLP(nn.Module):
    def __init__(
        self,
        config,
        input_dim,
        output_dim,
        hid_dims=[],
        last_activation=False,
        activation="relu",
    ):
        super().__init__()
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        elif acitvation == "elu":
            activation_fn = nn.Elu()
        else:
            return NotImplementedError

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        fc = []
        prev_dim = input_dim
        for d in hid_dims:
            fc.append(nn.Linear(prev_dim, d))
            fanin_init(fc[-1].weight)
            fc[-1].bias.data.fill_(0.1)
            fc.append(activation_fn)
            prev_dim = d
        fc.append(nn.Linear(prev_dim, output_dim))
        fc[-1].weight.data.uniform_(-1e-3, 1e-3)
        fc[-1].bias.data.uniform_(-1e-3, 1e-3)
        if last_activation:
            fc.append(activation_fn)
        self.fc = nn.Sequential(*fc)

    def forward(self, ob):
        return self.fc(ob)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

'''
ResNet Model Taken from https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html#resnet18
'''
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, robot_state, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear((512 * block.expansion) + robot_state, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, num_classes)
        self.relu_fc = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, robot_state):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)

        x = self.fc_1(x)
        x = self.relu_fc(x)
        x = self.fc_2(x)
        x = self.relu_fc(x)
        x = self.fc_3(x)

        return x

    def forward(self, x, robot_state):
        return self._forward_impl(x, robot_state)


def _resnet(arch, block, layers, pretrained, progress, robot_state, num_classes, **kwargs):
    model = ResNet(block, layers, robot_state, num_classes, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=False, robot_state=0, num_classes=1000, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, robot_state,
                   num_classes, **kwargs)

'''
MobileNetV3 from https://github.com/pytorch/vision/blob/ced96a0ca66e35ae61b64267e083fc3e3a62172a/torchvision/models/mobilenetv3.py#L267
'''

from torch import nn, Tensor
from functools import partial

def _make_divisible(v: float, divisor: int, min_value = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer = None,
        activation_layer = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation

class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer,
                 se_layer = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):

    def __init__(
            self,
            inverted_residual_setting,
            last_channel: int,
            robot_state,
            num_classes,
            block = None,
            norm_layer = None
    ) -> None:
        """
        MobileNet V3 main class
        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # original code
        # self.classifier = nn.Sequential(
        #     nn.Linear(lastconv_output_channels, last_channel),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(last_channel, num_classes),
        # )

        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels + robot_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, robot_state) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor, robot_state) -> Tensor:
        return self._forward_impl(x, robot_state)


def _mobilenet_v3_conf(arch: str, params):
    # non-public config parameters
    reduce_divider = 2 if params.pop('_reduced_tail', False) else 1
    dilation = 2 if params.pop('_dilated', False) else 1
    width_mult = params.pop('_width_mult', 1.0)

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting,
    last_channel: int,
    pretrained: bool,
    progress: bool,
    robot_state,
    num_classes,
    **kwargs
):
    model = MobileNetV3(inverted_residual_setting, last_channel, robot_state, num_classes, **kwargs)
    # if pretrained:
    #     if model_urls.get(arch, None) is None:
    #         raise ValueError("No checkpoint is available for model type {}".format(arch))
    #     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #     model.load_state_dict(state_dict)
    return model

def mobilenet_v3_small(pretrained: bool = False, progress: bool = False, robot_state = 0, num_classes = 1000, **kwargs) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, robot_state, num_classes, **kwargs)

'''
Model based on Shagun's implementation and https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
'''
class SimpleCNN(nn.Module):   
    def __init__(self, robot_state=0, num_classes=256):
        super(SimpleCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128 + robot_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass    
    def forward(self, x, robot_state):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)
        x = self.linear_layers(x)
        return x

class BC_Visual_Policy(nn.Module):   
    def __init__(self, robot_state=0, num_classes=256, img_size=128, device=None, env=None):
        super(BC_Visual_Policy, self).__init__()

        if device:
            self.device = device
        if env:
            self.env = env

        first_linear_layer_size = int(256 * math.floor(img_size / 8) * math.floor(img_size / 8))

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(first_linear_layer_size + robot_state, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass    
    def forward(self, x, robot_state):        
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)
        x = self.linear_layers(x)
        return x

    def visualize_filters(self, x):
        data = torch.from_numpy(x).cuda()
        out = self.cnn_layers[0](data)
        out = self.cnn_layers[1](out)
        out = self.cnn_layers[2](out)
        out = self.cnn_layers[3](out)
        return out

    def load_pretrained(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        filtered_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if self.state_dict()[k].size() != v.shape:
                print('load_pretrained_weights shape mismatch: model size {} ; pretrained size {}'.format(self.state_dict()[k].size(), v.shape))
                continue
            filtered_dict[k] = v
        self.load_state_dict(filtered_dict, strict=False)
        return None

    def load_bc_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

        # freeze all weights
        for param in self.parameters():
            param.requires_grad = False

        return None

    def get_predicted_ac(self, ob):
        if self.env == 'PusherObstacle-v0':
            inp = list(ob.values())
            inp_robot_state = inp[:2]
            if len(inp[0].shape) == 1: 
                inp_robot_state = [x.unsqueeze(0) for x in inp_robot_state]
            inp_robot_state = torch.cat(inp_robot_state, dim=-1)
            inp_img = inp[5]         
        elif 'Sawyer' in self.env:
            if len(ob['joint_pos'].shape) == 1:
                inp_robot_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat']])
                inp_robot_state = inp_robot_state[None, :]
                inp_img = ob['image']
            elif len(ob['joint_pos'].shape) == 2:
                inp_robot_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat']], axis=1)
                inp_img = ob['image']
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if len(inp_img.shape) == 5:
            inp_img = inp_img.squeeze(1) # remove unnecessary dimension
        out = self.forward(inp_img, inp_robot_state)
        if len(out.shape) == 3 and out.shape[1] == 1:
            out = out[:, 0, :]
        elif len(out.shape) == 2 and out.shape[0] == 1:
            out = out[0]

        return out

    def act_expert(self, ob):
        ob_copy = ob.copy()
        if 'image' in ob.keys() and isinstance(ob['image'], str):
            ob_copy['image'] = np.load(ob_copy['image'])
        ob_copy = to_tensor(ob_copy, self.device)
        actions = self.get_predicted_ac(ob_copy)
        ob_copy.clear()
        return actions

class BC_Visual_Policy_Stochastic(nn.Module): 
    def __init__(self, robot_state=0, num_classes=256, img_size=128, device=None, env=None):
        super(BC_Visual_Policy_Stochastic, self).__init__()

        if device:
            self.device = device
        if env:
            self.env = env

        first_linear_layer_size = int(256 * math.floor(img_size / 8) * math.floor(img_size / 8))
 
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(first_linear_layer_size + robot_state, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )

        self._activation_fn = getattr(F, 'relu')
    
        self.fc_means = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        self.fc_log_stds = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass    
    def forward(self, x, robot_state):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)
        x = self.linear_layers(x)

        x = self._activation_fn(x)

        means = self.fc_means(x)
        log_std = self.fc_log_stds(x)
        log_std = torch.clamp(log_std, -10, 2)
        stds = torch.exp(log_std.double())
        means = OrderedDict([("default", means)])
        stds = OrderedDict([("default", stds)])
        
        z = FixedNormal(means['default'], stds['default']).rsample()

        action = torch.tanh(z)
        
        return action

    def load_bc_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

        # freeze all weights
        for param in self.parameters():
            param.requires_grad = False

        return None

    def get_predicted_ac(self, ob):
        if self.env == 'PusherObstacle-v0':
            inp = list(ob.values())
            inp_robot_state = inp[:2]
            if len(inp[0].shape) == 1: 
                inp_robot_state = [x.unsqueeze(0) for x in inp_robot_state]
            inp_robot_state = torch.cat(inp_robot_state, dim=-1)
            inp_img = inp[5]         
        elif 'Sawyer' in self.env:
            if len(ob['joint_pos'].shape) == 1:
                inp_robot_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat']])
                inp_robot_state = inp_robot_state[None, :]
                inp_img = ob['image']
            elif len(ob['joint_pos'].shape) == 2:
                inp_robot_state = torch.cat([ob['joint_pos'], ob['joint_vel'], ob['gripper_qpos'], ob['gripper_qvel'], ob['eef_pos'], ob['eef_quat']], axis=1)
                inp_img = ob['image']
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if len(inp_img.shape) == 5:
            inp_img = inp_img.squeeze(1) # remove unnecessary dimension
        out = self.forward(inp_img, inp_robot_state)
        if len(out.shape) == 3 and out.shape[1] == 1:
            out = out[:, 0, :]
        elif len(out.shape) == 2 and out.shape[0] == 1:
            out = out[0]

        return out
        
    def act_expert(self, ob):
        ob_copy = ob.copy()
        if 'image' in ob.keys() and isinstance(ob['image'], str):
            ob_copy['image'] = np.load(ob_copy['image'])
        ob_copy = to_tensor(ob_copy, self.device)
        actions = self.get_predicted_ac(ob_copy)
        ob_copy.clear()
        return actions