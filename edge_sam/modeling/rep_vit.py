import torch.nn as nn
from .common import LayerNorm2d, UpSampleLayer, OpSequential, DynamicTanh

__all__ = ['rep_vit_m1', 'rep_vit_m2', 'rep_vit_m3', 'rep_vit_m1_dyt', 'RepViT']

# this is actually repvit_0_9
# https://github.com/THU-MIG/RepViT/blob/298f42075eda5d2e6102559fad260c970769d34e/model/repvit.py#L277C5-L277C16
m1_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 48, 1, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 96, 0, 0, 2],
    [3, 2, 96, 1, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 192, 0, 1, 2],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 384, 0, 1, 2],
    [3, 2, 384, 1, 1, 1],
    [3, 2, 384, 0, 1, 1]
]

m2_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 128, 0, 0, 2],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 256, 0, 1, 2],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 512, 0, 1, 2],
    [3, 2, 512, 1, 1, 1],
    [3, 2, 512, 0, 1, 1]
]

m3_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 128, 0, 0, 2],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 256, 0, 1, 2],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 512, 0, 1, 2],
    [3, 2, 512, 1, 1, 1],
    [3, 2, 512, 0, 1, 1]
]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from timm.models.layers import SqueezeExcite

import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        # self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        # adding from https://github.com/THU-MIG/RepViT/blob/main/model/repvit.py
        self.bn = torch.nn.BatchNorm2d(ed)

    # def forward(self, x):
    #     return self.conv(x) + self.conv1(x) + x
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
    """
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv
    """

class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, skip_downsample=False, gelu_approx='none'):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            if skip_downsample:
                stride = 1
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU(gelu_approx) if use_hs else nn.GELU(gelu_approx),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(gelu_approx) if use_hs else nn.GELU(gelu_approx),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


from timm.models.vision_transformer import trunc_normal_


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepViT(nn.Module):
    arch_settings = {
        'm1': m1_cfgs,
        'm2': m2_cfgs,
        'm3': m3_cfgs
    }

    def __init__(
        self, 
        arch, 
        img_size=1024, 
        fuse=False, 
        freeze=False,
        load_from=None, 
        use_rpn=False, 
        out_indices=None, 
        upsample_mode='bicubic',
        gelu_approx='none',
        dyt=False,
        return_low_feats=False,
        feature_names = ['stem', 'stage0']
    ):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = self.arch_settings[arch]
        self.img_size = img_size
        self.fuse = fuse
        self.freeze = freeze
        self.use_rpn = use_rpn
        self.out_indices = out_indices
        self.gelu_approx = gelu_approx
        self.dyt = dyt
        self.return_low_feats = return_low_feats
        self.feature_names = feature_names

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(gelu_approx),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            skip_downsample = False
            if not self.fuse and c in [384, 512]:
                skip_downsample = True
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, skip_downsample, gelu_approx))
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)

        if self.fuse:
            stage2_channels = _make_divisible(self.cfgs[self.stage_idx[2]][2], 8)
            stage3_channels = _make_divisible(self.cfgs[self.stage_idx[3]][2], 8)
            self.fuse_stage2 = nn.Conv2d(stage2_channels, 256, kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential([
                nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode=upsample_mode),
            ])
            neck_in_channels = 256
        else:
            neck_in_channels = output_channel
        
        if dyt:
            self.neck = nn.Sequential(
                nn.Conv2d(neck_in_channels, 256, kernel_size=1, bias=False),
                DynamicTanh(256, False),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                DynamicTanh(256, False),
            )
        else:
            self.neck = nn.Sequential(
                nn.Conv2d(neck_in_channels, 256, kernel_size=1, bias=False),
                LayerNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(256),
            )

        if load_from is not None:
            state_dict = torch.load(load_from)['model']
            new_state_dict = dict()
            use_new_dict = False
            for key in state_dict:
                if key.startswith('image_encoder.'):
                    use_new_dict = True
                    new_key = key[len('image_encoder.'):]
                    new_state_dict[new_key] = state_dict[key]
            if use_new_dict:
                state_dict = new_state_dict
            print(self.load_state_dict(state_dict, strict=True))

    def train(self, mode=True):
        super(RepViT, self).train(mode)
        if self.freeze:
            self.features.eval()
            self.neck.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        counter = 0
        output_dict = dict()
        # patch_embed
        x = self.features[0](x)
        # expect torch.Size([1, 48, 256, 256])
        output_dict['stem'] = x
        # stages
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f'stage{counter}'] = x
                counter += 1

        if self.fuse:
            x = self.fuse_stage2(output_dict['stage2']) + self.fuse_stage3(output_dict['stage3'])

        x = self.neck(x)
        output_dict['final'] = x

        if self.use_rpn:
            if self.out_indices is None:
                self.out_indices = [len(output_dict) - 1]
            out = []
            for i, key in enumerate(output_dict):
                if i in self.out_indices:
                    out.append(output_dict[key])
            return tuple(out)
        
        if self.return_low_feats:
            feats =  tuple([output_dict[n] for n in self.feature_names])
            return x, feats # output_dict['stem']
            
        return x


def rep_vit_m1(img_size=1024, **kwargs):
    return RepViT('m1', img_size, **kwargs)


def rep_vit_m2(img_size=1024, **kwargs):
    return RepViT('m2', img_size, **kwargs)


def rep_vit_m3(img_size=1024, **kwargs):
    return RepViT('m3', img_size, **kwargs)

def rep_vit_m1_dyt(img_size=1024, **kwargs):
    return RepViT('m1', img_size, dyt=True, gelu_approx='tanh', **kwargs)

# def rep_vit_m1_dyt_hq(img_size=1024, **kwargs):
#     return RepViT('m1', img_size, dyt=True, gelu_approx='tanh', return_low_feats=True, **kwargs)