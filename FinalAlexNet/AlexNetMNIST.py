import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from dependencies import value
from torch.nn import BatchNorm2d, BatchNorm1d
from torch.nn import Module

# Define bit widths and quantization type for binary
_activ_bits = 1
_weight_bits = 1
_input_bits = 8
_channels = [1, 96, 96, 128, 128, 96, 512, 64, 10]  # Changed first channel to 1 for grayscale input


class AlexNet_1W1A(Module):
    def __init__(self):
        super(AlexNet_1W1A, self).__init__()

        # Define layers
        self.conv1 = qnn.QuantConv2d(
            in_channels=1, out_channels=96, kernel_size=3, padding=1, bias=False,
            weight_quant=CommonWeightQuant, weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)
        self.norm1 = BatchNorm2d(96, eps=1e-4)
        self.act1 = qnn.QuantReLU(act_quant=CommonActQuant, bit_width=_activ_bits)

        self.conv2 = qnn.QuantConv2d(
            in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False,
            weight_quant=CommonWeightQuant, weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)
        self.norm2 = BatchNorm2d(96, eps=1e-4)
        self.act2 = qnn.QuantReLU(act_quant=CommonActQuant, bit_width=_activ_bits)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = qnn.QuantConv2d(
            in_channels=96, out_channels=128, kernel_size=3, padding=1, bias=False,
            weight_quant=CommonWeightQuant, weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)
        self.norm3 = BatchNorm2d(128, eps=1e-4)
        self.act3 = qnn.QuantReLU(act_quant=CommonActQuant, bit_width=_activ_bits)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = qnn.QuantConv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False,
            weight_quant=CommonWeightQuant, weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)
        self.norm4 = BatchNorm2d(128, eps=1e-4)
        self.act4 = qnn.QuantReLU(act_quant=CommonActQuant, bit_width=_activ_bits)

        self.conv5 = qnn.QuantConv2d(
            in_channels=128, out_channels=96, kernel_size=3, padding=1, bias=False,
            weight_quant=CommonWeightQuant, weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)
        self.norm5 = BatchNorm2d(96, eps=1e-4)
        self.act5 = qnn.QuantReLU(act_quant=CommonActQuant, bit_width=_activ_bits)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # We need to calculate the correct input size for fc1 dynamically
        self.flatten_dim = self._get_flatten_dim()

        # Fully connected layers
        self.fc1 = qnn.QuantLinear(
            self.flatten_dim, 512, bias=True, weight_quant=CommonWeightQuant,
            weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)
        self.norm_fc1 = BatchNorm1d(512, eps=1e-4)
        self.act_fc1 = qnn.QuantReLU(act_quant=CommonActQuant, bit_width=_activ_bits)

        self.fc2 = qnn.QuantLinear(
            512, 64, bias=True, weight_quant=CommonWeightQuant,
            weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)
        self.norm_fc2 = BatchNorm1d(64, eps=1e-4)
        self.act_fc2 = qnn.QuantReLU(act_quant=CommonActQuant, bit_width=_activ_bits)

        # Final output layer with 10 classes
        self.fc3 = qnn.QuantLinear(
            64, 10, bias=False, weight_quant=CommonWeightQuant,
            weight_bit_width=_weight_bits, quant_type=QuantType.BINARY)

    def _get_flatten_dim(self):
        # Create a dummy input tensor to compute the output size after convolutions
        dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input size: 1x28x28
        out = self.conv1(dummy_input)
        out = self.pool1(self.act2(self.norm2(self.conv2(out))))
        out = self.pool2(self.act3(self.norm3(self.conv3(out))))
        out = self.pool3(self.act5(self.norm5(self.conv5(out))))
        return out.view(1, -1).size(1)  # Flatten the output and return the size

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.pool1(self.act2(self.norm2(self.conv2(out))))
        out = self.pool2(self.act3(self.norm3(self.conv3(out))))
        out = self.act4(self.norm4(self.conv4(out)))
        out = self.pool3(self.act5(self.norm5(self.conv5(out))))
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.act_fc1(self.norm_fc1(self.fc1(out)))
        out = self.act_fc2(self.norm_fc2(self.fc2(out)))
        out = self.fc3(out)
        return out


class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1
    quant_type = QuantType.BINARY


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0
