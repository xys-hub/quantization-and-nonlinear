import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import math
import sys
sys.path.append("..")
from .quant import *

def quant_conv_relu(input, conv_layer, relu_layer):
    """
    Quantized Conv-ReLU layer 

    Args:
        input      (torch.tensor   ): quantized input tensor
        conv_layer (torch.nn.Conv2d): conv layer to extract the weight and bias
        relu_layer (torch.nn.ReLU  ): relu layer
    """
    
    conv_w  , conv_b   = conv_layer.weight.data, conv_layer.bias.data
    q_conv_w, q_conv_b = quant.WeightQuant(conv_w, conv_b, 'conv')
    out                = F.conv2d(input, q_conv_w, q_conv_b, stride=(1,1), padding=(1,1)) # self.conv3
    out                = quant.ActQuant(out, 'conv')
    out                = relu_layer(out)
    
    return out

def quant_fc_relu_dropout(input, fc_layer, relu_layer, dropout_layer):
    """
    Quantized FC-ReLU-Dropout layer 

    Args:
        input         (torch.tensor    ): quantized input tensor
        conv_layer    (torch.nn.Linear ): fc layer to extract the weight and bias
        relu_layer    (torch.nn.ReLU   ): relu layer
        dropout_layer (torch.nn.Dropout): dropout layer
    """
    
    fc_w  , fc_b   = fc_layer.weight, fc_layer.bias
    q_fc_w, q_fc_b = quant.WeightQuant(fc_w, fc_b, 'linear')
    out            = F.linear(input, q_fc_w, q_fc_b) # self.fc
    out            = quant.ActQuant(out, 'linear')
    out            = relu_layer(out)
    out            = dropout_layer(out)
    
    return out

def quant_fc(input, fc_layer):
    """
    Quantized fc_layer layer 

    Args:
        input         (torch.tensor    ): quantized input tensor
        conv_layer    (torch.nn.Linear ): fc layer to extract the weight and bias
    """
    
    fc_w  , fc_b   = fc_layer.weight, fc_layer.bias
    q_fc_w, q_fc_b = quant.WeightQuant(fc_w, fc_b, 'linear')
    out            = F.linear(input, q_fc_w, q_fc_b) # self.fc
    # out            = quant.ActQuant(out, 'linear')
    
    return out


def quant_conv_bn_relu(input, conv_layer, bn_layer, relu_layer):
    """
    Quantized Conv-BatchNorm-ReLU layer
    
    Args:
        input      (torch.tensor        ): 
        conv_layer (torch.nn.Conv2d     ): 
        bn_layer   (torch.nn.BatchNorm2d): 
        relu_layer (torch.nn.ReLU       ): 
    """
    
    q_conv_w, q_conv_b = ConvBNTrans(conv_layer, bn_layer)
    out                = F.conv2d(input, q_conv_w, q_conv_b, stride=(1,1), padding=(1,1)) # self.conv3
    out                = quant.ActQuant(out, 'conv')
    out                = relu_layer(out)
    
    
def ConvBNTrans(conv_layer, bn_layer):
    """
    Combine Conv2d with BatchNorm2d layer

    Args:
        conv_layer (nn.Conv2d     ): convolution layer 
        bn_layer   (nn.BatchNorm2d): batchnormalization layer
    """

    delta        = torch.sqrt(bn_layer.running_var + bn_layer.eps)
    foldScale    = torch.div(bn_layer.weight, delta)
    # print('foldScale size: ', foldScale.size())
    # print('weight size: ', conv_layer.weight.size())
    foldedWeight = torch.mul(conv_layer.weight, 
                            torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(foldScale, 1), 2), 3))
    if conv_layer.bias is not None:
        foldedBias   = bn_layer.bias + conv_layer.bias - torch.div(torch.mul(bn_layer.weight, bn_layer.running_mean), delta)
    else:
        foldedBias   = bn_layer.bias - torch.div(torch.mul(bn_layer.weight, bn_layer.running_mean), delta)
                            
    return foldedWeight, foldedBias

