# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author : Minahil Raza


from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d
from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.zero_point import ZeroZeroPoint
from .tensor_norm import TensorNorm
from .common import CommonWeightQuant, CommonActQuant

_activ_bits = 8
_weight_bits = 8
_input_bits = 8
_quant_type=QuantType.INT
_channels = [3, 96, 96, 128, 128, 96, 512, 64, 10]

class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        #Initial quantization of input data to ensure 8 bits.
        self.linear = qnn.QuantIdentity(act_quant=CommonActQuant,
                                        bit_width=_input_bits,
                                        min_val=- 1.0,
                                        max_val=1.0 - 2.0 ** (-7),
                                        narrow_range=False,
                                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO)
        
        
        
        #First two convolutions, activations, and pooling layer
        self.conv1 = qnn.QuantConv2d(in_channels= _channels[0],
                                     out_channels= _channels[1],
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.norm1 = BatchNorm2d(_channels[1],
                                 eps=1e-4)
        

        self.relu1 = qnn.QuantIdentity(act_quant=CommonActQuant,
                                   bit_width=_activ_bits)
        

        self.conv2 = qnn.QuantConv2d(in_channels= _channels[1],
                                     out_channels= _channels[2],
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.norm2 = BatchNorm2d(_channels[2],
                                 eps=1e-4)

        self.relu2 = qnn.QuantIdentity(act_quant=CommonActQuant,
                                   bit_width=_activ_bits)


        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        
        
        
        #Third convolution, activation, and second pooling layer
        self.conv3 = qnn.QuantConv2d(in_channels= _channels[2],
                                     out_channels= _channels[3],
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.norm3 = BatchNorm2d(_channels[3],
                                 eps=1e-4)

        self.relu3 = qnn.QuantIdentity(act_quant=CommonActQuant,
                                   bit_width=_activ_bits)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        
        
        
        #Forth and fifth convolution, activation, and third pooling layer
        self.conv4 = qnn.QuantConv2d(in_channels= _channels[3],
                                     out_channels= _channels[4],
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.norm4 = BatchNorm2d(_channels[3],
                                 eps=1e-4)

        self.relu4 = qnn.QuantIdentity(act_quant=CommonActQuant,
                                   bit_width=_activ_bits)

        self.conv5 = qnn.QuantConv2d(in_channels= _channels[4],
                                     out_channels= _channels[5],
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.norm5 = BatchNorm2d(_channels[5],
                                 eps=1e-4)

        self.relu5 = qnn.QuantIdentity(act_quant=CommonActQuant,
                                   bit_width=_activ_bits)


        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        # -----------------------------------------------------------------------
        
        
        #First full convolution linear layer - note this is linear!
        self.fc1   = qnn.QuantLinear(4*4*_channels[5], _channels[6],
                                     bias= True,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.normfc1 = BatchNorm1d(_channels[6],
                                 eps=1e-4)
        

        self.relufc1 = qnn.QuantIdentity(act_quant=CommonActQuant,
                                   bit_width=_activ_bits)

        
        
        #Second full convolution linear layer
        self.fc2   = qnn.QuantLinear(_channels[6], _channels[7],
                                     bias= True,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.normfc2 = BatchNorm1d(_channels[7],
                                 eps=1e-4)

        self.relufc2 = qnn.QuantIdentity(act_quant=CommonActQuant,
                                   bit_width=_activ_bits)

        
        
        #Third and final convolution linear layer. Output must be 10 for 10 classes. Bias should be FALSE.
        self.fc3   = qnn.QuantLinear(_channels[7], _channels[8],
                                     bias= False,
                                     weight_quant=CommonWeightQuant,
                                     weight_bit_width=_weight_bits,
                                     quant_type=_quant_type)
        
        self.finalnorm = TensorNorm()
        
        

        
    def clip_weights(self, min_val, max_val):
        self.conv1.weight.data.clamp_(min_val, max_val)
        self.conv2.weight.data.clamp_(min_val, max_val)
        self.conv3.weight.data.clamp_(min_val, max_val)
        self.conv4.weight.data.clamp_(min_val, max_val)
        self.conv5.weight.data.clamp_(min_val, max_val)
        self.fc1.weight.data.clamp_(min_val, max_val)
        self.fc2.weight.data.clamp_(min_val, max_val)
        

    def forward(self, x):
        out = self.linear(x)
        
        out = self.relu1(self.norm1(self.conv1(out)))
        #print("post:",out)
        out = self.relu2(self.norm2(self.conv2(out)))
        out = self.pool1(out)

        out = self.relu3(self.norm3(self.conv3(out)))
        out = self.pool2(out)

        out = self.relu4(self.norm4(self.conv4(out)))
        out = self.relu5(self.norm5(self.conv5(out)))
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.relufc1(self.normfc1(self.fc1(out)))
        out = self.relufc2(self.normfc2(self.fc2(out)))
        out = self.fc3(out)
        
        out = self.finalnorm(out)
        
        #print("pre:",out)
        #out = F.log_softmax(out, dim=1)
        return out
