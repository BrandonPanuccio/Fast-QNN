import torch.nn as nn

import brevitas.nn as qnn

class QuantAlexNet(nn.Module):
    """
    A slightly adapted AlexNet-like architecture for 28x28 MNIST images.
    Uses Brevitas quantization layers (QuantConv2d, QuantLinear, QuantReLU).
    """

    def __init__(self, num_bits=8, num_classes=10):
        super(QuantAlexNet, self).__init__()

        # Instead of specifying explicit quant module wrappers, you can directly pass
        # weight_bit_width=num_bits and act_bit_width=num_bits to Brevitas layers.
        # We'll do it this way for simplicity:

        self.features = nn.Sequential(
            # 1st Conv
            qnn.QuantConv2d(
                in_channels=1, out_channels=64, kernel_size=11, stride=4, padding=2,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd Conv
            qnn.QuantConv2d(
                in_channels=64, out_channels=192, kernel_size=5, padding=2,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3rd Conv
            qnn.QuantConv2d(
                in_channels=192, out_channels=384, kernel_size=3, padding=1,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),

            # 4th Conv
            qnn.QuantConv2d(
                in_channels=384, out_channels=256, kernel_size=3, padding=1,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),

            # 5th Conv
            qnn.QuantConv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits)
            # We omit the final max-pool from classic AlexNet because it would collapse MNIST's 1x1 to 0x0.
        )

        # After the 5th convolution, the spatial dimension is (likely) 1x1 => flattened dimension = 256
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            qnn.QuantLinear(256, 4096, weight_bit_width=num_bits, bias=False),
            qnn.QuantReLU(bit_width=num_bits),

            nn.Dropout(p=0.5),
            qnn.QuantLinear(4096, 4096, weight_bit_width=num_bits, bias=False),
            qnn.QuantReLU(bit_width=num_bits),

            qnn.QuantLinear(4096, num_classes, weight_bit_width=num_bits, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x