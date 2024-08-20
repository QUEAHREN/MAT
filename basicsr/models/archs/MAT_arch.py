import math
from torch.autograd import Variable
import einops
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import numpy as np
from einops import rearrange
import torchvision
import cv2


def get_non_zero_ratio(x: torch.Tensor) -> torch.Tensor:
    ''' Get the ratio of non-zero elements in each bin for four SAST blocks'''
    '''Input: (B, C, H, W). Output: [(B, C), (B, C), (B, C), (B, C)].'''
    # Downsample to match the receptive field of each SAST block.
    x_down_2 = torch.nn.functional.max_pool2d(x.float(), kernel_size=2, stride=2)
    x_down_4 = torch.nn.functional.max_pool2d(x_down_2, kernel_size=2, stride=2)
    # Count the number of non-zero elements in each bin.
    num_nonzero_1 = torch.sum(torch.sum(x != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_2 = torch.sum(torch.sum(x_down_2 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_3 = torch.sum(torch.sum(x_down_4 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)

    result1 = x.shape[0] / x.numel() * num_nonzero_1.float()
    result2 = x.shape[0] / x_down_2.numel() * num_nonzero_2.float()
    result3 = x.shape[0] / x_down_4.numel() * num_nonzero_3.float()
    # Return the ratio of non-zero elements in each bin at four scales.
    return [abs(result1), abs(result2), abs(result3)]


class PositiveLinear(nn.Module):
    ''' Linear layer with positive weights'''
    def __init__(self, in_features, out_features, bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Apply exponential function to ensure weights are positive
        positive_weights = torch.exp(self.weight)
        return nn.functional.linear(input, positive_weights, self.bias)

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')



##########################################################################
## Layer Norm
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def ChannelSplit(inp):
    C = inp.shape[1]
    return inp[:, :C//2, :, :], inp[:, C//2:, :, :]

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # q1 = self.CS(q)
        # k1 = self.CS(k)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # visualize_attention_matrix(attn.squeeze(0).squeeze(0).data.cpu())
        attn = attn.softmax(dim=-1)


        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class MSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.deconv = DeformConv2D(dim, dim*3, 3, 1)

        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # self.para_proj = nn.Linear(, )


    def forward(self, x, mask):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q = q*mask
        k = k*mask
       
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)

        return out
    


class MAA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MAA, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        self.temperature_e = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
       
    # origin
    def forward(self, x, ME):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        in_ME = -1 * (ME - 1)
        q_enhe = ME * self.temperature_e + in_ME
        q = q*q_enhe

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        del q,k
        attn = attn.softmax(dim=-1)

        out = attn @ v
        del v,attn
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out

   
class CMIG(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(CMIG, self).__init__()

        # g0
        self.dim = dim
        hidden_features = dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def sort_feature(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        avg_values = avg_pool.view(x.size(0), x.size(1))
        sorted_indices = avg_values.argsort(dim=1)
        # print(sorted_indices)
        x[:,:,:,:] = x[:, sorted_indices,:,:]
        return x

    def forward(self, x, ev):
        # g0
        x = F.gelu(ev) * x
        x = self.project_out(x)
        return x


class TransformerBlock_MAT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, use_evs=False, evs_enc=False):
        super(TransformerBlock_MAT, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if not evs_enc:
            if not use_evs:
                self.attn = Attention(dim, num_heads, bias)
            else:
                self.attn = MAA(dim, num_heads, bias)
        else:
            self.attn = MSA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        
        self.use_evs = use_evs
        if use_evs:
            self.fusion = CMIG(dim, ffn_expansion_factor, bias)
            self.norm0 = LayerNorm(dim, LayerNorm_type)
            self.norm_evs = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, Mask_E=None):

        if self.use_evs:
            C = x.shape[1]
            # print(x.shape)
            events = x[:, C//2:, :, :]
            x = x[:, :C//2, :, :]
            
            x = self.fusion(self.norm0(x), self.norm_evs(events)) + x
        if Mask_E is None:
            x = x + self.attn(self.norm1(x))
        else:
            x = x + self.attn(self.norm1(x), Mask_E)
        # save_as_heatmap(x[:, 0, :, :], f"xvis_{x.size(2)}_{1}")
        
        x = x + self.ffn(self.norm2(x))
        if self.use_evs:
            return torch.cat([x, events], 1), Mask_E  
        else: 
            return x
        
            

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
    

def topk2D(
    input: torch.Tensor,
    k: torch.Tensor,
    largest: torch.Tensor
) -> torch.Tensor:
    max_k = k.max().int()
    # print(k)
    # input = input.squeeze(1).squeeze(1)
    fake_indexes = torch.topk(input, max_k, dim=-1, largest=largest).indices
    # torch.cuda.synchronize()
    T = torch.arange(max_k).expand_as(fake_indexes).to(device=input.device)
    T = torch.remainder(T, k.unsqueeze(1))
    
    indexes = torch.gather(fake_indexes, -1, T)
    # print(indexes.shape)
    return indexes

# import faiss
class AMMP(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim, LayerNorm_type):
        super().__init__()

        self.in_norm = LayerNorm(embed_dim, LayerNorm_type)
        self.bins = 6
        self.to_controls = PositiveLinear(self.bins, embed_dim*2, bias=False)

        # conv head (chd)
        self.in_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.GELU()
        )

        self.conv_compress = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim//2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim//2, 2, bias=False),
            nn.GELU()
        )
        self.channel_compress = nn.Sequential(
            ChannelPool(),
            nn.Conv2d(2, 2, kernel_size=1, bias=False),
            nn.GELU()
        )

        self.in_compress3 = nn.Sequential(
            nn.Linear(4, 1, bias=False),
            nn.GELU()
        )
        # self.out_proj = nn.LogSoftmax(dim=-1)
        
        self.alpha = nn.Parameter(torch.ones(1, 1, 1)*0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask, ratio):
        B, C, H, W = x.size()
        x_= x
        x = self.in_norm(x)
        # conv head (chd)
        local_x = self.in_proj(x)

        ratio = ratio[:, None, None, :] # [B,1,1,Bin]
        scale = self.to_controls(ratio)

        # Global Agg
        mask = rearrange(mask, "b head c (h w) -> b (head c) h w", head=1, h=H, w=W)
        global_x = (local_x * mask).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True) / (torch.sum(torch.sum(mask, dim=-1, keepdim=True), dim=-2, keepdim=True))
        global_x[global_x==torch.inf] = 0
        # Channel Compress
        x_1 = self.channel_compress(local_x*global_x)
        x_1 = rearrange(x_1, "b (i c) h w -> b i (h w) c", c=2, h=H, w=W)

        x_2 = torch.cat([local_x, global_x.expand(B, C, H, W)], dim=1) * scale.permute(0, -1, 1, 2)
        x_2 = rearrange(x_2, "b (i c) h w -> b i (h w) c", c=2*C, h=H, w=W)
        x_2 = self.conv_compress(x_2)
        
        x = self.in_compress3(torch.cat([x_1, x_2], dim=-1))
        output = rearrange(x, "b (head c) h w -> b head c (h w)", head=1) 

        # -------------------topk2d_for training-------------------
        # mu_r = torch.mean(ratio, dim=-1) 
        # b = torch.ones(mu_r.shape, device=output.device)*0.4
        # m = mu_r/self.alpha
        # K = torch.where(m >= b, b, m)
        # K = torch.where(K <= mu_r, mu_r, K)
        # K = torch.where(K <= b*0.01, 0.005, K)
        # K = K*H*W
        # K = K.int()
        # print(self.alpha)
        # indexs = topk2D(output, k=K, largest=True)
        # new_mask = torch.zeros(B, 1, 1, H*W, device=x.device, requires_grad=False)
        # new_mask.scatter_(-1, indexs, 1.)
        # indexs = topk2D(output, k=K, largest=False)
        # new_mask.scatter_(-1, indexs, 1.)
        # -------------------topk2d_for traning-------------------

        # ratio_scale (rs)   alpha == beta in paper
        self.alpha = ratio.max() if self.alpha < ratio.max() else self.alpha
        m = ratio.max()/self.alpha
        K = m if m<=0.4 else 0.4
        K = torch.where(K <= 0.004, 0.005, K)

        # topk_maxr for testing
        indexs = torch.topk(output, k=int(K*H*W), dim=-1, largest=True, sorted=False)[1]
        new_mask = torch.zeros(B, 1, 1, H*W, device=x.device, requires_grad=False)
        new_mask.scatter_(-1, indexs, 1.)
        indexs = torch.topk(output, k=int(K*H*W), dim=-1, largest=False, sorted=False)[1]
        new_mask.scatter_(-1, indexs, 1.)
        
        # (rs)6
        return new_mask, self.sigmoid(self.alpha)*F.gelu(rearrange(x, "b i (h w) c -> b (i c) h w", c=1, h=H, w=W))
    

class UNetEVTransformerBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, num_heads):
        super(UNetEVTransformerBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0, bias=True)
    
        self.score_predictor = AMMP(out_size, LayerNorm_type="WithBias")
        self.encoder = [TransformerBlock_MAT(out_size, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias", use_evs=False, evs_enc=True) for _ in range(2)]
        self.encoder = nn.Sequential(*self.encoder)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, ratio=None, merge_before_downsample=True):

        B, C, H, W = x.size()
        prev_decision = torch.ones(B, 1, 1, H*W, dtype=x.dtype, device=x.device)  
        out = self.conv_1(x)
        
                
        for i, enc in enumerate(self.encoder):
            pred_score, weighting = self.score_predictor(out, prev_decision, ratio)
            mask = pred_score
            #  -------------ECSG------------
            out = out*weighting + out
            #  -----------------------------
            out = enc(out, mask) 
            prev_decision = mask

        out = out + self.identity(x)

        if self.downsample:
            out_down = self.downsample(out)         
            if not merge_before_downsample:   
                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out, mask
        else:
            out = self.conv_before_merge(out)
            return out, mask


class CustomSequential(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x1, x2 = x
            x = module(x1, x2)
        return x
    
class MAT(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=[8, 8, 7],
        num_refinement_blocks=2,
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(MAT, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**0),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True
                )
                for i in range(num_blocks[0])
            ]
        ) 
  

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True
                )
                for i in range(num_blocks[1])
            ]
        ) 
      

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
  
        self.encoder_level3 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True
                )
                for i in range(num_blocks[2])
            ]
        ) 
   

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )


        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )
        self.down_path_ev = nn.ModuleList()
        prev_channels = dim
        depth = len(num_blocks)
        self.depth = depth
        for i in range(depth):
            downsample = True if (i+1) < depth else False 
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(UNetEVTransformerBlock(prev_channels, (2**i) * dim, downsample,num_heads=heads[i]))

            prev_channels = (2**i) * dim

        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.conv_ev1 = nn.Conv2d(6, dim, 3, 1, 1)
        self.fuse_before_downsample = True

        
    def forward(self, x):
        
        inp_img = x[:,0:3,:,:]
        events = x[:,3:9,:,:]
        ratio = get_non_zero_ratio(events) # [B, bin]

        ev = []
        se = []
        e1 = self.conv_ev1(events)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up, score = down(e1, ratio[i], self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1, score = down(e1, ratio[i], self.fuse_before_downsample)
                ev.append(e1)
            se.append(score)


        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = torch.cat([inp_enc_level1, ev[0]], 1)
        # print(inp_enc_level1.shape)
        out_enc_level1, _ = self.encoder_level1((inp_enc_level1, se[0]))
        out_enc_level1 = ChannelSplit(out_enc_level1)[0]
        del inp_enc_level1
        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2 = torch.cat([inp_enc_level2, ev[1]], 1)
        out_enc_level2, _ = self.encoder_level2((inp_enc_level2, se[1]))
        out_enc_level2 = ChannelSplit(out_enc_level2)[0]
        del inp_enc_level2
        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc_level3 = torch.cat([inp_enc_level3, ev[2]], 1)
        
        out_enc_level3, _ = self.encoder_level3((inp_enc_level3, se[2]))
        inp_dec_level3 = ChannelSplit(out_enc_level3)[0]

        del ev, out_enc_level3, inp_enc_level3
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        del out_enc_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        del out_enc_level1
        out_dec_level1 = self.refinement(out_dec_level1)


        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


from calflops import calculate_flops
# pip install calflops transformers
if __name__ == "__main__":
    model = MAT()
    # print(model)
    # input = torch.randn(1, 9, 128, 128)
    # output = model(input)
    # print("-" * 50)
    # print(output.shape)

    # from thop import profile
    # flops, params = profile(model, inputs=torch.randn(1, 1, 9, 1024, 728))
    # print(" FLOPs:%s    Params:%s \n" %(flops, params))


    input_shape = (1, 9, 1024, 728)
    flops, macs, params = calculate_flops(
        model=model, input_shape=input_shape, output_as_string=True, output_precision=4
    )
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
