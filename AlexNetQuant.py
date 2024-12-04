import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantMaxPool2d, QuantIdentity
from common import CommonWeightQuant, CommonActQuant

class AlexNetQuant(nn.Module):
    def __init__(self, num_classes=1000, weight_bit_width=1, act_bit_width=1):
        super(AlexNetQuant, self).__init__()

        weight_quant = CommonWeightQuant
        act_quant = CommonActQuant

        self.features = nn.Sequential(
            QuantIdentity(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True),
            QuantConv2d(3, 64, kernel_size=11, stride=4, padding=2, weight_quant=weight_quant,
                        bit_width=weight_bit_width, bias=False),
            QuantReLU(act_quant=act_quant, bit_width=act_bit_width),
            QuantMaxPool2d(kernel_size=3, stride=2),

            QuantConv2d(64, 192, kernel_size=5, padding=2, weight_quant=weight_quant, bit_width=weight_bit_width,
                        bias=False),
            QuantReLU(act_quant=act_quant, bit_width=act_bit_width),
            QuantMaxPool2d(kernel_size=3, stride=2),

            QuantConv2d(192, 384, kernel_size=3, padding=1, weight_quant=weight_quant, bit_width=weight_bit_width,
                        bias=False),
            QuantReLU(act_quant=act_quant, bit_width=act_bit_width),

            QuantConv2d(384, 256, kernel_size=3, padding=1, weight_quant=weight_quant, bit_width=weight_bit_width,
                        bias=False),
            QuantReLU(act_quant=act_quant, bit_width=act_bit_width),

            QuantConv2d(256, 256, kernel_size=3, padding=1, weight_quant=weight_quant, bit_width=weight_bit_width,
                        bias=False),
            QuantReLU(act_quant=act_quant, bit_width=act_bit_width),
            QuantMaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            QuantLinear(256 * 6 * 6, 4096, weight_quant=weight_quant, bit_width=weight_bit_width, bias=False),
            QuantReLU(act_quant=act_quant, bit_width=act_bit_width),

            QuantLinear(4096, 4096, weight_quant=weight_quant, bit_width=weight_bit_width, bias=False),
            QuantReLU(act_quant=act_quant, bit_width=act_bit_width),

            QuantLinear(4096, num_classes, weight_quant=weight_quant, bit_width=weight_bit_width, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_quant(num_classes=1000, weight_bit_width=1, act_bit_width=1):
    return AlexNetQuant(num_classes=num_classes, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)


# Example usage
if __name__ == "__main__":
    model = alexnet_quant(num_classes=1000, weight_bit_width=1, act_bit_width=1)
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(output.shape)
