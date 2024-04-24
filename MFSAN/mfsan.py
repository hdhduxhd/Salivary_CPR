import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import copy
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from cpr.tool import pyutils

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def dice_ce_loss(pred, target):
    smooth = 1e-5  # 平滑项，用于避免分母为零
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum(1)
    union = m1.sum(1) + m2.sum(1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()
    ce = F.binary_cross_entropy(pred, target, reduction='mean')
    return dice_loss + ce

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(32 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*2, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*2, cout=width*2, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*2, cout=width*4, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))
        
        self.aff_cup = torch.nn.Conv2d(width*8, width*8, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.aff_cup.weight, gain=4)
        self.bn_cup = nn.BatchNorm2d(width*8)
        self.bn_cup.weight.data.fill_(1)
        self.bn_cup.bias.data.zero_()

        self.from_scratch_layers = [self.aff_cup, self.bn_cup]
        
        image_res = 512
        radius = 4
        self.predefined_featuresize = int(image_res//16)
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=radius, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        
        feature = x
        f_cup = F.relu(self.bn_cup(self.aff_cup(feature)))###bn

        if f_cup.size(2) == self.predefined_featuresize and f_cup.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            print('featuresize error')
            sys.exit()

        f_cup = f_cup.view(f_cup.size(0), f_cup.size(1), -1)

        ff = torch.index_select(f_cup, dim=2, index=ind_from.to(x.device))
        ft = torch.index_select(f_cup, dim=2, index=ind_to.to(x.device))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff_cup = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))
        return x, features[::-1], aff_cup

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size=224, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        grid_size = (16,16)
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        if patch_size[0] == 0:
            patch_size = (1, 1)

        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=768,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)


    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
  
def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)

        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
#         cos_layer = self.transpose_for_scores(hidden_states)
        
#         dot_product = torch.matmul(cos_layer, cos_layer.transpose(-1,-2))
#         norm = torch.norm(cos_layer, dim=-1).unsqueeze(-1)
#         norm_matrix = torch.matmul(norm, norm.transpose(-1,-2))
#         cos_matrix = torch.div(dot_product, norm_matrix)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_scores = torch.mul(attention_scores, cos_matrix)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
    
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    
class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        #layers
        for _ in range(12):
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, img_size=224, in_channels=3, vis=False):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights

class Conv2dReLU(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel_size,padding=0,stride=1,use_batchnorm=True,):
        conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=not (use_batchnorm),)
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,skip_channels=0,use_batchnorm=True,):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, skip=None):
#         print("x.shape before up:",x.shape)
        x = self.up(x)
#         print("x.shape after up:",x.shape)
        if skip is not None:
#             print("skip.shape:",skip.shape)
            x = torch.cat([x, skip], dim=1)
#             print("x.shape after cat:",x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class DecoderCup(nn.Module):
    def __init__(self,img_size=512):
        super().__init__()
        head_channels = 256
        self.conv_more = Conv2dReLU(
            768,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = (128, 64, 32, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        sf = img_size//16//16
        if sf==0:
            sf = 1
        self.up = nn.UpsamplingBilinear2d(scale_factor=sf)

        if 3 != 0:
            skip_channels = [128, 64, 32, 16]
            for i in range(4-3):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) 
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.up(x)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < 3) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class Segmentation(nn.Module):
    def __init__(self, zero_head=False):
        super(Segmentation, self).__init__()
        self.zero_head = zero_head
        self.classifier = 'seg'
        self.decoder = DecoderCup()
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
        )
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x, features):
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = torch.sigmoid(self.bn(logits))
        return logits

from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

loss_domain = nn.CrossEntropyLoss()

class MFSAN(nn.Module):

    def __init__(self, img_size=224):
        super(MFSAN, self).__init__()
        self.sharedNet = ResNetV2((3, 4, 9), 1)
        self.sonnet1 = Transformer(img_size=img_size, in_channels=256)
        self.sonnet2 = Transformer(img_size=img_size, in_channels=256)
        self.GRL=GRL()
        self.discriminator1=nn.Sequential(
            nn.Linear(16*16*768,100),
            nn.BatchNorm1d(100),
            nn.GELU(),
            nn.Linear(100,2),
            nn.Softmax(dim=1)
        )
        self.discriminator2=nn.Sequential(
            nn.Linear(16*16*768,100),
            nn.BatchNorm1d(100),
            nn.GELU(),
            nn.Linear(100,2),
            nn.Softmax(dim=1)
        )
        self.cls_fc_son1 = Segmentation()
        self.cls_fc_son2 = Segmentation()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1, alpha = 1):
        batch_size = data_src.shape[0]
        mmd_loss = 0
        if self.training == True:
            if mark == 1:
                data_src, data_src_feature, _ = self.sharedNet(data_src)
                data_tgt, data_tgt_feature, _ = self.sharedNet(data_tgt)

                data_tgt_son1 = self.sonnet1(data_tgt)[0]

                data_src = self.sonnet1(data_src)[0]
                
                src_domain_output = self.discriminator1(GRL.apply(data_src.view(data_src.size(0), -1),alpha))
                src_domain_label = torch.zeros(batch_size, device=data_src.device)
                src_domain_label = src_domain_label.long()
                
                # mmd_loss += loss_domain(src_domain_output, src_domain_label)
                
                tgt_domain_output = self.discriminator1(GRL.apply(data_tgt_son1.view(data_tgt_son1.size(0), -1),alpha))
                tgt_domain_label = torch.ones(batch_size, device=data_src.device)
                tgt_domain_label = tgt_domain_label.long()
                # mmd_loss += loss_domain(tgt_domain_output, tgt_domain_label)
                
                # mmd_loss += mmd(data_src.view(data_src.size(0), -1), data_tgt_son1.view(data_tgt_son1.size(0), -1))

                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1, data_tgt_feature)

                data_tgt_son2 = self.sonnet2(data_tgt)[0]
                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2, data_tgt_feature)

                l1_loss = torch.abs(data_tgt_son1.view(data_tgt_son1.size(0), -1)
                                    - data_tgt_son2.view(data_tgt_son2.size(0), -1))
                l1_loss = torch.mean(l1_loss)
                pred_src = self.cls_fc_son1(data_src, data_src_feature)

                cls_loss = dice_ce_loss(pred_src, label_src)

                return cls_loss, mmd_loss, l1_loss

            if mark == 2:
                data_src, data_src_feature, _ = self.sharedNet(data_src)
                data_tgt, data_tgt_feature, _ = self.sharedNet(data_tgt)

                data_tgt_son2 = self.sonnet2(data_tgt)[0]

                data_src = self.sonnet2(data_src)[0]
                
                src_domain_output = self.discriminator2(GRL.apply(data_src.view(data_src.size(0), -1),alpha))
                src_domain_label = torch.zeros(batch_size, device=data_src.device)
                src_domain_label = src_domain_label.long()
                # mmd_loss += loss_domain(src_domain_output, src_domain_label)
                
                tgt_domain_output = self.discriminator2(GRL.apply(data_tgt_son2.view(data_tgt_son2.size(0), -1),alpha))
                tgt_domain_label = torch.ones(batch_size, device=data_src.device)
                tgt_domain_label = tgt_domain_label.long()
                # mmd_loss += loss_domain(tgt_domain_output, tgt_domain_label)
                
#                 mmd_loss += mmd(data_src.view(data_src.size(0), -1), data_tgt_son2.view(data_tgt_son2.size(0), -1))

                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2, data_tgt_feature)

                data_tgt_son1 = self.sonnet1(data_tgt)[0]
                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1, data_tgt_feature)
                
                l1_loss = torch.abs(data_tgt_son1.view(data_tgt_son1.size(0), -1)
                                    - data_tgt_son2.view(data_tgt_son2.size(0), -1))
                l1_loss = torch.mean(l1_loss)

                #l1_loss = F.l1_loss(torch.nn.functional.softmax(data_tgt_son1, dim=1), torch.nn.functional.softmax(data_tgt_son2, dim=1))

                pred_src = self.cls_fc_son2(data_src, data_src_feature)
                cls_loss = dice_ce_loss(pred_src, label_src)

                return cls_loss, mmd_loss, l1_loss

        else:
            data1, feature1, _ = self.sharedNet(data_src)
            data2, feature2, _ = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data1)[0]
            pred1 = self.cls_fc_son1(fea_son1, feature1)

            fea_son2 = self.sonnet2(data2)[0]
            pred2 = self.cls_fc_son2(fea_son2, feature2)

            return pred1, pred2
