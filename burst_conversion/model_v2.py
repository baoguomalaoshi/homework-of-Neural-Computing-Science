from torch import nn
import torch
import math
from torch.nn.utils import fusion


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)



    def forward(self, x):
        # batched_imgs = x[0].new_full([x.shape[0],x.shape[1],224,320], 0)
        # for img, pad_img in zip(x, batched_imgs):
        #     # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
        #     # 这样保证输入到网络中一个batch的每张图片的shape相同
        #     # copy_: Copies the elements from src into self tensor and returns self
        #     pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # x = batched_imgs

        # for i in self.features:
        #     x = i(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # def forward(self, x):
    #     fuse_list = []
    #     self.eval()
    #     with torch.no_grad():  # conv bn融合
    #         for i in self.features:
    #             if type(i) == ConvBNReLU:
    #                 fuse_list.append(fusion.fuse_conv_bn_eval(i[0], i[1]))
    #                 fuse_list.append(nn.ReLU6(inplace=True))
    #             else:
    #                 for j in range(len(i.conv)):
    #                     print(i.conv[j])
    #                     if type(i.conv[j]) == torch.nn.Conv2d:
    #                         fuse_list.append(fusion.fuse_conv_bn_eval(i.conv[j], i.conv[j + 1]))
    #                     elif type(i.conv[j]) == ConvBNReLU:
    #                         fuse_list.append(fusion.fuse_conv_bn_eval(i.conv[j][0], i.conv[j][1]))
    #                         fuse_list.append(nn.ReLU6(inplace=True))
    #     fuse_list = nn.Sequential(*fuse_list)
    #     print(fuse_list)
    #     self.fuse_list = fuse_list
    #     x = fuse_list[0](x)
    #     x = fuse_list[1](x)
    #     x = fuse_list[2](x)
    #     x = fuse_list[3](x)
    #     x = fuse_list[4](x)
    #     x = fuse_list[5](x)
    #     x = fuse_list[6](x)
    #     x = fuse_list[7](x)
    #     x = fuse_list[8](x)
    #     x = fuse_list[9](x)
    #     x = fuse_list[10](x)
    #     x = fuse_list[11](x)
    #     x = fuse_list[12](x)
    #     x = fuse_list[13](x)
    #     x = fuse_list[14](x)
    #     x = fuse_list[15](x)
    #     x = fuse_list[16](x)
    #     x = fuse_list[17](x)
    #     x = fuse_list[18](x)
    #     x = fuse_list[19](x)
    #     x = fuse_list[20](x)
    #     x = fuse_list[21](x)
    #     x = fuse_list[22](x)
    #     x = fuse_list[23](x)
    #     x = fuse_list[24](x)
    #     x = fuse_list[25](x)
    #     x = fuse_list[26](x)
    #     x = fuse_list[27](x)
    #     x = fuse_list[28](x)
    #     x = fuse_list[29](x)
    #     x = fuse_list[30](x)
    #     x = fuse_list[31](x)
    #     x = fuse_list[32](x)
    #     x = fuse_list[33](x)
    #     x = fuse_list[34](x)
    #     x = fuse_list[35](x)
    #     x = fuse_list[36](x)
    #     x = fuse_list[37](x)
    #     x = fuse_list[38](x)
    #     x = fuse_list[39](x)
    #     x = fuse_list[40](x)
    #     x = fuse_list[41](x)
    #     x = fuse_list[42](x)
    #     x = fuse_list[43](x)
    #     x = fuse_list[44](x)
    #     x = fuse_list[45](x)
    #     x = fuse_list[46](x)
    #     x = fuse_list[47](x)
    #     x = fuse_list[48](x)
    #     x = fuse_list[49](x)
    #     x = fuse_list[50](x)
    #     x = fuse_list[51](x)
    #     x = fuse_list[52](x)
    #     x = fuse_list[53](x)
    #     x = fuse_list[54](x)
    #     x = fuse_list[55](x)
    #     x = fuse_list[56](x)
    #     x = fuse_list[57](x)
    #     x = fuse_list[58](x)
    #     x = fuse_list[59](x)
    #     x = fuse_list[60](x)
    #     x = fuse_list[61](x)
    #     x = fuse_list[62](x)
    #     x = fuse_list[63](x)
    #     x = fuse_list[64](x)
    #     x = fuse_list[65](x)
    #     x = fuse_list[66](x)
    #     x = fuse_list[67](x)
    #     x = fuse_list[68](x)
    #     x = fuse_list[69](x)
    #     x = fuse_list[70](x)
    #     x = fuse_list[71](x)
    #     x = fuse_list[72](x)
    #     x = fuse_list[73](x)
    #     x = fuse_list[74](x)
    #     x = fuse_list[75](x)
    #     x = fuse_list[76](x)
    #     x = fuse_list[77](x)
    #     x = fuse_list[78](x)
    #     x = fuse_list[79](x)
    #     x = fuse_list[80](x)
    #     x = fuse_list[81](x)
    #     x = fuse_list[82](x)
    #     x = fuse_list[83](x)
    #     x = fuse_list[84](x)
    #     x = fuse_list[85](x)
    #     x = fuse_list[86](x)
    #
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     return x
