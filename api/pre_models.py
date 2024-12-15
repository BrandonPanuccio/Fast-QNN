import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType

class QuantAlexNet(nn.Module):
    """
    A slightly adapted AlexNet-like architecture for e.g. MNIST-like inputs.
    Uses Brevitas quantization layers (QuantConv2d, QuantLinear, QuantReLU).
    Explicitly sets 'weight_quant_type' and 'quant_type' so that FINN can parse them.
    Layer-by-layer structure (no nn.Sequential).
    """
    def __init__(self, num_bits=1, num_classes=10):
        super(QuantAlexNet, self).__init__()

        # Decide on quantization type based on bit-width
        if num_bits == 1:
            weight_quant_type = QuantType.BINARY
            act_quant_type = QuantType.BINARY
        else:
            weight_quant_type = QuantType.INT
            act_quant_type = QuantType.INT

        # ---------------------------
        #  CONVOLUTIONAL FEATURES
        # ---------------------------

        # 1st Conv block
        self.conv1 = qnn.QuantConv2d(
            in_channels=1, out_channels=64, kernel_size=11, stride=4, padding=2,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )
        self.act1 = qnn.QuantReLU(bit_width=num_bits, quant_type=act_quant_type)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd Conv block
        self.conv2 = qnn.QuantConv2d(
            in_channels=64, out_channels=192, kernel_size=5, padding=2,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )
        self.act2 = qnn.QuantReLU(bit_width=num_bits, quant_type=act_quant_type)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd Conv block
        self.conv3 = qnn.QuantConv2d(
            in_channels=192, out_channels=384, kernel_size=3, padding=1,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )
        self.act3 = qnn.QuantReLU(bit_width=num_bits, quant_type=act_quant_type)

        # 4th Conv block
        self.conv4 = qnn.QuantConv2d(
            in_channels=384, out_channels=256, kernel_size=3, padding=1,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )
        self.act4 = qnn.QuantReLU(bit_width=num_bits, quant_type=act_quant_type)

        # 5th Conv block
        self.conv5 = qnn.QuantConv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )
        self.act5 = qnn.QuantReLU(bit_width=num_bits, quant_type=act_quant_type)
        # Skipping final MaxPool because of the small input size (28x28).

        # ---------------------------
        #    CLASSIFIER LAYERS
        # ---------------------------
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = qnn.QuantLinear(
            256, 4096,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )
        self.act_fc1 = qnn.QuantReLU(bit_width=num_bits, quant_type=act_quant_type)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = qnn.QuantLinear(
            4096, 4096,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )
        self.act_fc2 = qnn.QuantReLU(bit_width=num_bits, quant_type=act_quant_type)

        self.fc3 = qnn.QuantLinear(
            4096, num_classes,
            weight_bit_width=num_bits, weight_quant_type=weight_quant_type, bias=False
        )

    def forward(self, x):
        # 1st Conv block
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        # 2nd Conv block
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # 3rd Conv block
        x = self.conv3(x)
        x = self.act3(x)

        # 4th Conv block
        x = self.conv4(x)
        x = self.act4(x)

        # 5th Conv block
        x = self.conv5(x)
        x = self.act5(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.act_fc1(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.act_fc2(x)

        x = self.fc3(x)
        return x