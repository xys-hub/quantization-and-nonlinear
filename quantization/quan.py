import torch
import math

sparse = 0
true_sparse = 0
number = 0
MaskedConv2d_flag = 0

def mask(x, size=(1,1), maxpool=2):
    if maxpool == 3:
        AvgPool = nn.AvgPool2d(3,2,1)
    elif maxpool == 2:
        AvgPool = nn.AvgPool2d(2,2,0)
    Upsample = nn.Upsample(scale_factor=2, mode='nearest')
    x = torch.mean(x,dim=1,keepdim=True)
    #print("size = ",end = "")
    #print(size)
    if size == 1:#1X1
        mask_flag0 = (x >= x.mean()/2).float()
        #mask_flag0 = (x >= 25/255).float()
        mask_flag1 = AvgPool(mask_flag0)
        mask_flag2 = AvgPool(mask_flag1)
        mask_flag3 = AvgPool(mask_flag2)
        mask_flag4 = AvgPool(mask_flag3)
        mask_flag5 = AvgPool(mask_flag4)
    elif size == 2:#2X2
        x0 = AvgPool(x)
        mask_flag1 = (x0 >= x0.mean()/2).float()
        mask_flag0 = Upsample(mask_flag1)
        mask_flag2 = AvgPool(mask_flag1)
        mask_flag3 = AvgPool(mask_flag2)
        mask_flag4 = AvgPool(mask_flag3)
        mask_flag5 = AvgPool(mask_flag4)
    elif size == 4:#4X4
        x0 = AvgPool(x)
        x0 = AvgPool(x0)
        mask_flag2 = (x0 >= x0.mean()/2).float()
        mask_flag1 = Upsample(mask_flag2)
        mask_flag0 = Upsample(mask_flag1)
        mask_flag3 = AvgPool(mask_flag2)
        mask_flag4 = AvgPool(mask_flag3)
        mask_flag5 = AvgPool(mask_flag4)
    elif size == 8:#8X8
        x0 = AvgPool(x)
        x0 = AvgPool(x0)
        x0 = AvgPool(x0)
        mask_flag3 = (x0 >= x0.mean()/2).float()
        mask_flag2 = Upsample(mask_flag3)
        mask_flag1 = Upsample(mask_flag2)
        mask_flag0 = Upsample(mask_flag1)
        mask_flag4 = AvgPool(mask_flag3)
        mask_flag5 = AvgPool(mask_flag4)
    elif size == 16:#16X16
        x0 = AvgPool(x)
        x0 = AvgPool(x0)
        x0 = AvgPool(x0)
        x0 = AvgPool(x0)
        mask_flag4 = (x0 >= x0.mean()/2).float()
        mask_flag3 = Upsample(mask_flag4)
        mask_flag2 = Upsample(mask_flag3)
        mask_flag1 = Upsample(mask_flag2)
        mask_flag0 = Upsample(mask_flag1)
        mask_flag5 = AvgPool(mask_flag4)
    elif size == 32:#32X32
        x0 = AvgPool(x)
        x0 = AvgPool(x0)
        x0 = AvgPool(x0)
        x0 = AvgPool(x0)
        x0 = AvgPool(x0)
        mask_flag5 = (x0 >= x0.mean()/2).float()
        mask_flag4 = Upsample(mask_flag5)
        mask_flag3 = Upsample(mask_flag4)
        mask_flag2 = Upsample(mask_flag3)
        mask_flag1 = Upsample(mask_flag2)
        mask_flag0 = Upsample(mask_flag1)
    else:
        AvgPool_img = nn.AvgPool2d(size)
        Upsample_img = nn.Upsample(x[0][0].shape, mode='nearest')
        #print(x.shape)
        x = AvgPool_img(x)
        #print(x.shape)
        x = Upsample_img(x)
        #print(x.shape)
        mask_flag0 = (x >= x.mean()/1.5).float()
        mask_flag1 = AvgPool(mask_flag0)
        mask_flag2 = AvgPool(mask_flag1)
        mask_flag3 = AvgPool(mask_flag2)
        mask_flag4 = AvgPool(mask_flag3)
        mask_flag5 = AvgPool(mask_flag4)

    global sparse
    if mask_layer == 0:
        sparse_single = (mask_flag0.sum()*18+mask_flag1.sum()*360.25+mask_flag2.sum()*3856+mask_flag3.sum()*41328+mask_flag4.sum()*147312+mask_flag5.sum()*608112)/2164.109375/x.shape[2]/x.shape[3]
    elif mask_layer == 1:
        sparse_single = (416*416*18+mask_flag0.sum()*360.25+mask_flag1.sum()*3856+mask_flag2.sum()*41328+mask_flag3.sum()*147312+mask_flag4.sum()*608112)/2164.109375/x.shape[2]/x.shape[3]
    elif mask_layer == 2:
        sparse_single = (416*416*18+208*208*360.25+mask_flag0.sum()*3856+mask_flag1.sum()*41328+mask_flag2.sum()*147312+mask_flag3.sum()*608112)/2164.109375/x.shape[2]/x.shape[3]
    
    
    #print(sparse_single)
    sparse += sparse_single/116
    return mask_flag0[0][0], mask_flag1[0][0], mask_flag2[0][0], mask_flag3[0][0], mask_flag4[0][0], mask_flag5[0][0]

def get_fl(max_value, quant_bit):
    if max_value>0:
        il = math.ceil(math.log(max_value, 2))
        fl = quant_bit -1 - il
        return fl
    else:
        return 0

def quantzie(x, fl, quant_bit):
    # Saturate data
    max_data = (math.pow(2, quant_bit - 1) -1) * math.pow(2,-fl)
    min_data = -(math.pow(2, quant_bit - 1)) * math.pow(2, -fl)
    x = torch.clamp(x, min_data, max_data)
    # round
    x = torch.div(x,math.pow(2, -fl))
    x = torch.round(x)
    x = torch.mul(x,math.pow(2,-fl))
    return x

# def zjm_round(value):
    # tmp_a = (value >= 0)
    # pos_value = tmp_a * (value + 0.5)
    # pos_quan = pos_value.int()

    # return pos_quan

def zjm_round(value):
    tmp_a = (value >= 0)
    tmp_b = (value < 0)
    pos_value = tmp_a * (value + 0.5)
    pos_quan = pos_value.int()

    neg_value = tmp_b * (value - 0.5)
    neg_quan = neg_value.int()

    tmp_c = (torch.abs(torch.abs(neg_quan - tmp_b * value) - 0.5) <= 1e-6)
    neg_quan = neg_quan + tmp_c
    return neg_quan + pos_quan

def quantzie_act(x, fl, quant_bit):
    # Saturate data
    #print(fl)
    max_data = (math.pow(2, quant_bit - 1) -1) * math.pow(2,-fl)
    min_data = -(math.pow(2, quant_bit - 1)) * math.pow(2, -fl)
    x = torch.clamp(x, min_data, max_data)
    # round
    x = torch.div(x,math.pow(2, -fl))
    x = zjm_round(x)
    x = torch.mul(x,math.pow(2,-fl))
    return x


def quan_weight(model, W_BIT_WIDTH=8):
    for i, named_parameter in enumerate(model.named_parameters()):
        name, parameters=named_parameter
        # print(i, name)
        new_weight_data = parameters.data.clone()
        # if name == "module_list.22.Conv2d.weight":
        #     new_weight_data[1+8][227][0][0] = 0.5
        max_value = torch.max(torch.abs(new_weight_data)).cpu().detach().numpy()
        fl=get_fl(max_value, W_BIT_WIDTH)
        # if name == "module_list.22.Conv2d.weight":
        #     print("max_value = ", max_value)
        #     print("fl = ", fl)
        #     print(new_weight_data[1+8][227][0][0])
        new_weight_data=quantzie(new_weight_data, fl, W_BIT_WIDTH)
        # if name == "module_list.22.Conv2d.weight":
        #     print(new_weight_data[1+8][227][0][0])
        #print(i,(new_weight_data == 0).float().sum(),(new_weight_data < 0).float().sum(),(new_weight_data > 0).float().sum())
        parameters.data = new_weight_data

def quan_activation(output, index, mtype=None, A_BIT_WIDTH=8):
    max_value = torch.max(torch.abs(output)).cpu().detach().numpy()
    fl = get_fl(max_value, A_BIT_WIDTH)
    # shortcut_list1 = [2, 6, 15, 21, 40, 43, 46, 49, 52, 55, 56, 58, 61, 71, 74]
    # shortcut_list2 = [11, 18, 24, 27, 30, 36, 65]
    # shortcut_list3 = [33]
    #shortcut_list1 = [2, 6, 15, 21, 40, 43, 46, 49, 52, 55, 71, 74]
    #shortcut_list2 = [11, 18, 24, 27, 30, 36, 65]
    #shortcut_list3 = [33]
    # plus1_list = [59, 62, 66]
    #plus2_list = [89]
    #scale_list = {89:5, 90:6, 91:6, 92:6, 93:6, 94:4,}

    #if index in shortcut_list1:
    #    fl -= 1
    #elif index in shortcut_list2:
    #    fl -= 2
    #elif index in shortcut_list3:
    #    fl -= 3
    #elif index in scale_list:
    #    fl = scale_list[index]

    # elif index == 85:
    #     fl = 3
    #if index >= 80:
    #    print(index, fl, mtype)
        
    output = quantzie_act(output, fl, A_BIT_WIDTH)
    return output

def quan_activation_mask(output, index, mask_flag0, mask_flag1, mask_flag2, mask_flag3, mask_flag4, mask_flag5, mtype=None, A_BIT_WIDTH=8, low_WIDTH=8):
    max_value = torch.max(torch.abs(output)).cpu().detach().numpy()
    fl = get_fl(max_value, A_BIT_WIDTH)
    f2 = get_fl(max_value, low_WIDTH)
    if output[0][0].shape == mask_flag0.shape:
        mask_flag = mask_flag0
    elif output[0][0].shape == mask_flag1.shape:
        mask_flag = mask_flag1
    elif output[0][0].shape == mask_flag2.shape:
        mask_flag = mask_flag2
    elif output[0][0].shape == mask_flag3.shape:
        mask_flag = mask_flag3
    elif output[0][0].shape == mask_flag4.shape:
        mask_flag = mask_flag4
    elif output[0][0].shape == mask_flag5.shape:
        mask_flag = mask_flag5
    else:
        #print(index)
        #print(output[0][0].shape)
        mask_flag = 1

    output1 = quantzie_act(output, fl, A_BIT_WIDTH)
    output2 = quantzie_act(output, f2, low_WIDTH)
    if low_WIDTH == 0:
        output2 = 0
    output = output1 * mask_flag + output2 * (1 - mask_flag)
    return output

def FakeQuantize_static(x, WIDTH):
    max_value = torch.max(torch.abs(x))
    fl = get_fl(max_value, WIDTH)
    x1 = quantzie(x, fl, WIDTH)
    return x1

class FakeQuantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, WIDTH):
        max_value = torch.max(torch.abs(x))
        fl = get_fl(max_value, WIDTH)
        x1 = quantzie(x, fl, WIDTH)
        return x1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None