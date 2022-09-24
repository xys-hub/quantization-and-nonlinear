import torch
import torch.nn as nn
import copy

class QuantizeScheme(object):
    """
    Quantization scheme definition: Google INT8-Only Quantization
    """
    def __init__(self):
        self.scheme        = 'google'
        self.subscheme     = 'per_channel'
        self.weight_bits   = 8
        self.act_bits      = 8

quan_scheme = QuantizeScheme()


def WeightQuant(weight, bias, ConvOrLinear='conv', is_cuda=True):
    """
    Function for weight quantization

    Args:
        weight        (torch.tensor): weights for the module
        bias          (torch.tensor): bias for the module
        ConvOrLinear  (string      ): 'conv': Conv2D layer; 'linear': Linear layer
        is_cuda       (bool        ): bool variable for cuda array
    """
    
    if ConvOrLinear == 'conv':
        # ***** set the dimensions
        O, C, H, W = weight.size()
        # ***** create the array
        intMax      = torch.zeros([O,C,H,W], dtype=torch.float32).cuda() # int max values
        # intMin      = torch.zeros([O,C,H,W], dtype=torch.float32).cuda() # int min values
        # intRange    = torch.zeros([O,C,H,W], dtype=torch.float32).cuda() # int min values
        # ***** find the values
        TensorTwo   = torch.tensor(2, dtype=torch.float32).cuda()
        intMax.fill_(torch.pow(TensorTwo, quan_scheme.weight_bits-1) - 1) # 127 = 2^7 - 1
        intMin      = intMax * -1  # -127 = -1 * (2^7 - 1)
        intRange    = intMax - intMin # 127 - (-127) = 254
        fpMax       = torch.max(weight.view(weight.size()[0], -1), dim=1)[0] # First, reshape it to (N, C*H*W). Then, find the max value of ifmap(i,:,:,:)
        fpMin       = torch.min(weight.view(weight.size()[0], -1), dim=1)[0]
        fpMax       = fpMax.unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        fpMax       = fpMax.expand(O, C, H, W)
        fpMin       = fpMin.unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        fpMin       = fpMin.expand(O, C, H, W)
        fp2intScale = intRange / (fpMax - fpMin)
        # ***** quantize the weights and bias 
        weight.mul_(fp2intScale) # weight*int_range/fp_range
        weight.round_()          # round to int
        weight.div_(fp2intScale)
        if bias is not None:
            bias.mul_(fp2intScale[:,0,0,0].view(-1))
            bias.round_()
            bias.div_(fp2intScale[:,0,0,0].view(-1))
    else:
        # ***** create the array
        intMax      = torch.zeros([1], dtype=torch.float32).cuda() # int max values
        # intMin      = torch.zeros([1], dtype=torch.float32).cuda() # int min values
        # intRange    = torch.zeros([1], dtype=torch.float32).cuda() # int min values
        # ***** find the values
        TensorTwo   = torch.tensor(2, dtype=torch.float32).cuda()
        intMax.fill_(torch.pow(TensorTwo, quan_scheme.weight_bits-1) - 1) # 127 = 2^7 - 1
        intMin      = intMax * -1  # -127 = -1 * (2^7 - 1)
        intRange    = intMax - intMin # 127 - (-127) = 254
        fpMax       = torch.tensor(torch.max(weight).item()) # First, reshape it to (N, C*H*W). Then, find the max value of ifmap(i,:,:,:)
        fpMin       = torch.tensor(torch.min(weight).item())
        fpMax       = fpMax.cuda()
        fpMin       = fpMin.cuda()
        fp2intScale = intRange / (fpMax - fpMin)
        # ***** quantize the weights and bias 
        weight.mul_(fp2intScale) # weight*int_range/fp_range
        weight.round_()          # round to int
        weight.div_(fp2intScale)
        if bias is not None:
            bias.mul_(fp2intScale)
            bias.round_()
            bias.div_(fp2intScale)
    
    return weight, bias

def ActQuant(input, ConvOrLinear='conv', is_cuda=True):
    """
    Function for weight quantization

    Args:
        input        (torch.tensor)  : network input or ofmap of each conv/linear layer 
        ConvOrLinear (str, optional) : 'conv': Conv2D layer; 'linear': Linear layer. Defaults to 'conv'.
        is_cuda      (bool, optional): bool variable for cuda array. Defaults to True.

    Returns:
        ofmap_quan: quantized feature map
    """
    
    if ConvOrLinear == 'conv':
        # ***** set the dimensions
        N, C, H, W = input.size() # Batch_size, Channel_in, Height, Width
        # ***** create the array
        intMax      = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int max values
        # intMin      = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int min values
        # intRange    = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int min values
        # ***** find the values
        TensorTwo   = torch.tensor(2, dtype=torch.float32).cuda()
        intMax.fill_(torch.pow(TensorTwo, quan_scheme.act_bits-1) - 1) # 127 = 2^7 - 1
        intMin      = intMax * -1  # -127 = -1 * (2^7 - 1)
        intRange    = intMax - intMin # 127 - (-127) = 254
        fpMax       = torch.max(input.view(input.size()[0], input.size()[1], -1), dim=2)[0] # First, reshape it to (N, C*H*W). Then, find the max value of ifmap(i,:,:,:)
        fpMin       = torch.min(input.view(input.size()[0], input.size()[1], -1), dim=2)[0]
        fpMax       = fpMax.unsqueeze(2).unsqueeze(3).cuda()
        fpMin       = fpMin.unsqueeze(2).unsqueeze(3).cuda()
        fpMax       = fpMax.expand(N, C, H, W)
        fpMin       = fpMin.expand(N, C, H, W)
        fp2intScale = intRange / (fpMax - fpMin)
        # print('fpMax max, min:', torch.max(fpMax).item(), torch.min(fpMax).item())
        # print('fpMin max, min:', torch.max(fpMin).item(), torch.min(fpMin).item())
        # print('intRange max, min:', torch.max(intRange).item(), torch.min(intRange).item())
        # print('fpScale nan: ', torch.isnan(fp2intScale).any() )
        # print('fpScale[10]', fp2intScale[1:10,2,6,6])
        # print('fpScale max, min:', torch.max(fp2intScale).item(), torch.min(fp2intScale).item())
        # print('fp2intScale zeros: ', torch.any(fp2intScale == 0))
        # print('input[10]', input[1:10,2,6,6])
        # ***** quantize the inputs and bias 
        # input.mul_(fp2intScale) # input*int_range/fp_range
        input = torch.mul(input, fp2intScale) # input*int_range/fp_range
        # print('mul nan: ', torch.isnan(input).any() )
        input.round_()          # round to int
        input.div_(fp2intScale)
    else: 
        # ***** set the dimensions
        N, C = input.size() # Batch_size, Channel_in
        # ***** create the array
        intMax      = torch.zeros([N,C], dtype=torch.float32).cuda() # int max values
        # intMin      = torch.zeros([N,C], dtype=torch.float32).cuda() # int min values
        # intRange    = torch.zeros([N,C], dtype=torch.float32).cuda() # int min values
        # ***** find the values
        TensorTwo   = torch.tensor(2, dtype=torch.float32).cuda()
        intMax.fill_(torch.pow(TensorTwo, quan_scheme.act_bits-1) - 1) # 127 = 2^7 - 1
        intMin      = intMax * -1  # -127 = -1 * (2^7 - 1)
        intRange    = intMax - intMin # 127 - (-127) = 254
        fpMax       = torch.max(input, dim=1)[0] # First, reshape it to (N, C*H*W). Then, find the max value of ifmap(i,:,:,:)
        fpMin       = torch.min(input, dim=1)[0]
        fpMax       = fpMax.cuda()
        fpMin       = fpMin.cuda()
        fpMax       = fpMax.unsqueeze(1).expand(N, C)
        fpMin       = fpMin.unsqueeze(1).expand(N, C)
        fp2intScale = intRange / (fpMax - fpMin)
        # ***** quantize the inputs and bias 
        input.mul_(fp2intScale) # input*int_range/fp_range
        input.round_()          # round to int
        input.div_(fp2intScale)
    
    return input

def FixedActQuant(input, ConvOrLinear='conv', is_cuda=True):
    """
    Function for weight quantization

    Args:
        input        (torch.tensor)  : network input or ofmap of each conv/linear layer 
        ConvOrLinear (str, optional) : 'conv': Conv2D layer; 'linear': Linear layer. Defaults to 'conv'.
        is_cuda      (bool, optional): bool variable for cuda array. Defaults to True.

    Returns:
        ofmap_quan: quantized feature map
    """
    
    if ConvOrLinear == 'conv':
        # ***** set the dimensions
        N, C, H, W = input.size() # Batch_size, Channel_in, Height, Width
        # ***** create the array
        intMax      = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int max values
        fpMax       = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int max values
        fpMin       = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int max values
        # intMin      = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int min values
        # intRange    = torch.zeros([N,C,H,W], dtype=torch.float32).cuda() # int min values
        # ***** find the values
        TensorTwo   = torch.tensor(2, dtype=torch.float32).cuda()
        intMax.fill_(torch.pow(TensorTwo, quan_scheme.act_bits-1) - 1) # 127 = 2^7 - 1
        intMin      = intMax * -1  # -127 = -1 * (2^7 - 1)
        fpMax.fill_(6.0)
        fpMin.fill_(-6.0)
        intRange    = intMax - intMin # 127 - (-127) = 254
        fp2intScale = intRange / (fpMax - fpMin)
        # ***** quantize the inputs and bias 
        input.mul_(fp2intScale) # input*int_range/fp_range
        input.round_()          # round to int
        input.div_(fp2intScale)
    else: 
        # ***** set the dimensions
        N, C = input.size() # Batch_size, Channel_in
        # ***** create the array
        intMax      = torch.zeros([N,C], dtype=torch.float32).cuda() # int max values
        # intMin      = torch.zeros([N,C], dtype=torch.float32).cuda() # int min values
        # intRange    = torch.zeros([N,C], dtype=torch.float32).cuda() # int min values
        # ***** find the values
        TensorTwo   = torch.tensor(2, dtype=torch.float32).cuda()
        intMax.fill_(torch.pow(TensorTwo, quan_scheme.act_bits-1) - 1) # 127 = 2^7 - 1
        intMin      = intMax * -1  # -127 = -1 * (2^7 - 1)
        intRange    = intMax - intMin # 127 - (-127) = 254
        fpMax       = torch.max(input, dim=1)[0] # First, reshape it to (N, C*H*W). Then, find the max value of ifmap(i,:,:,:)
        fpMin       = torch.min(input, dim=1)[0]
        fpMax       = fpMax.cuda()
        fpMin       = fpMin.cuda()
        fpMax       = fpMax.unsqueeze(1).expand(N, C)
        fpMin       = fpMin.unsqueeze(1).expand(N, C)
        fp2intScale = intRange / (fpMax - fpMin)
        # ***** quantize the inputs and bias 
        input.mul_(fp2intScale) # input*int_range/fp_range
        input.round_()          # round to int
        input.div_(fp2intScale)
    
    return input
