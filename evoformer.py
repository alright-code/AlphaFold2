import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EvoFormer(nn.Module):
    def __init__(self, num_blocks=48, c_m=384, h_m=5, w_m=5, a_m=32, heads_m=8, c_z=128, h_z=5, w_z=5, a_z=128, heads_z=4, c_s=384):
        super().__init__()
        
        self.blocks = nn.ModuleList([Block(c_m=c_m, h_m=h_m, w_m=w_m, a_m=a_m, heads_m=heads_m,
                                           c_z=c_z, h_z=h_z, w_z=w_z, a_z=a_z, heads_z=heads_z) for _ in range(num_blocks)])
        
        self.w_s = nn.Conv1d(c_m, c_s, 1)
        
    def forward(self, msa_rep, pair_rep):
        for block in self.blocks:
            msa_rep, pair_rep = checkpoint(block, msa_rep, pair_rep)
        
        s = self.w_s(msa_rep[:, :, 0, :])
        
        return msa_rep, pair_rep, s

        
class Block(nn.Module):
    def __init__(self, c_m, h_m, w_m, a_m, heads_m, c_z, h_z, w_z, a_z, heads_z):
        super().__init__()
        
        # MSA Stack
        self.a1 = RowWiseGatedAttention(h_m=h_m, w_m=w_m, c_m=c_m, num_heads=heads_m, h_z=h_z, w_z=w_z, c_z=c_z, c=a_m)
        self.d1 = RowWiseDropout(0.15)
        self.a2 = ColWiseGatedAttention(h_m=h_m, w_m=w_m, c_m=c_m, num_heads=heads_m, c=a_m)
        self.trans1 = Transition(h=h_m, w=w_m, c=c_m)
        
        # Communication
        self.op_mean = OPMean(c_m=c_m, h_m=h_m, w_m=w_m, c_z=c_z, c=a_m)
        
        # Pair Stack
        self.tri1 = TriangleOutgoing(c_z=c_z, h_z=h_z, w_z=w_z, c=a_z)
        self.d2 = RowWiseDropout(0.25)
        self.tri2 = TriangleIncoming(c_z=c_z, h_z=h_z, w_z=w_z, c=a_z)
        self.tri3 = TriangleAttentionStart(c_z=c_z, h_z=h_z, w_z=w_z, num_heads=heads_z, c=a_z)
        self.tri4 = TriangleAttentionEnd(c_z=c_z, h_z=h_z, w_z=w_z, num_heads=heads_z, c=a_z)
        self.d3 = ColWiseDropout(0.25)
        self.trans2 = Transition(h=h_z, w=w_z, c=c_z)
        
    def forward(self, msa_rep, pair_rep):
        # MSA Stack
        msa_rep = msa_rep + self.d1(self.a1(msa_rep, pair_rep))
        msa_rep = msa_rep + self.a2(msa_rep)
        msa_rep = msa_rep + self.trans1(msa_rep)
        
        # Communication
        pair_rep = pair_rep + self.op_mean(msa_rep)
        
        # Pair Stack
        pair_rep = pair_rep + self.d2(self.tri1(pair_rep))
        pair_rep = pair_rep + self.d2(self.tri2(pair_rep))
        pair_rep = pair_rep + self.d2(self.tri3(pair_rep))
        pair_rep = pair_rep + self.d3(self.tri4(pair_rep))
        pair_rep = pair_rep + self.trans2(pair_rep)
        
        return msa_rep, pair_rep


class RowWiseGatedAttention(nn.Module):
    def __init__(self, h_m, w_m, c_m, num_heads, h_z, w_z, c_z, c):
        super().__init__()

        self.c = c
        self.num_heads = num_heads
        
        self.ln1 = nn.LayerNorm([c_m, h_m, w_m])
        self.w_q = nn.Conv2d(c_m, c * num_heads, 1, bias=False)
        self.w_k = nn.Conv2d(c_m, c * num_heads, 1, bias=False)
        self.w_v = nn.Conv2d(c_m, c * num_heads, 1, bias=False)
        
        self.ln2 = nn.LayerNorm([c_z, h_z, w_z])
        self.w_b = nn.Conv2d(c_z, num_heads, 1, bias=False)
        
        self.w_g = nn.Conv2d(c_m, c * num_heads, 1)
        
        self.w_rep = nn.Conv2d(c * num_heads, c_m, 1)
        
    def forward(self, msa_rep, pair_rep):
        # Input Projections
        import ipdb; ipdb.set_trace()
        msa_rep = self.ln1(msa_rep)                                                                                         # b x c_m x s x i
        q = self.w_q(msa_rep)                                                                                               # b x (c * h) x s x i
        k = self.w_k(msa_rep)                                                                                               # b x (c * h) x s x i
        v = self.w_v(msa_rep)                                                                                               # b x (c * h) x s x i
        
        b = self.w_b(self.ln2(pair_rep))                                                                                    # b x h x i x j
        
        g = torch.sigmoid(self.w_g(msa_rep))                                                                                # b x (c * h) x s x i
        g = einops.rearrange(g, 'b (h c) s i -> b c h s i', h=self.num_heads)                                               # b x c x h x s x i
        
        # Attention
        q = einops.rearrange(q, 'b (h c) s i -> b c h s i', h=self.num_heads)                                               # b x c x h x s x i
        k = einops.rearrange(k, 'b (h c) s i -> b c h s i', h=self.num_heads)                                               # b x c x h x s x i
        v = einops.rearrange(v, 'b (h c) s i -> b c h s i', h=self.num_heads)                                               # b x c x h x s x i
        
        a = torch.softmax(math.sqrt(self.c)**-1 * torch.einsum('bchsi,bchsj->bhsij', q, k) + b.unsqueeze(2), dim=-1)        # b x h x s x i x j
        a = einops.repeat(a, 'b h s i j -> b c h s i j', c=v.shape[1])                                                      # b x c x h x s x i x j
        v = einops.repeat(v, 'b c h s i -> b c h s i j', j=a.shape[-1])                                                     # b x c x h x s x i x j

        o = g * torch.einsum('bchsij,bchsij->bchsi', a, v)                                                                  # b x c x h x s x i
        
        # Output 
        o = einops.rearrange(o, 'b c h s i -> b (c h) s i')                                                                 # b x (c * h) x s x i
        msa_rep = self.w_rep(o)                                                                                             # b x c_m x s x i

        return msa_rep
        

class ColWiseGatedAttention(nn.Module):
    def __init__(self, h_m, w_m, c_m, num_heads, c):
        super().__init__()
        
        self.c = c
        self.num_heads = num_heads
        
        self.ln1 = nn.LayerNorm([c_m, h_m, w_m])
        self.w_q = nn.Conv2d(c_m, c * num_heads, 1, bias=False)
        self.w_k = nn.Conv2d(c_m, c * num_heads, 1, bias=False)
        self.w_v = nn.Conv2d(c_m, c * num_heads, 1, bias=False)
        
        self.w_g = nn.Conv2d(c_m, c * num_heads, 1)
        
        self.w_rep = nn.Conv2d(c * num_heads, c_m, 1)
        
    def forward(self, msa_rep):
        # Input projections
        msa_rep = self.ln1(msa_rep)                                                                         # b x c_m x s x i
        q = self.w_q(msa_rep)                                                                               # b x (c * h) x s x i
        k = self.w_k(msa_rep)                                                                               # b x (c * h) x s x i
        v = self.w_v(msa_rep)                                                                               # b x (c * h) x s x i
        
        g = torch.sigmoid(self.w_g(msa_rep))                                                                # b x (c * h) x s x i
        g = einops.rearrange(g, 'b (h c) s i -> b c h s i', h=self.num_heads)                               # b x c x h x s x i
        
        # Attention
        q = einops.rearrange(q, 'b (h c) s i -> b c h s i', h=self.num_heads)                               # b x c x h x s x i
        k = einops.rearrange(k, 'b (h c) s i -> b c h s i', h=self.num_heads)                               # b x c x h x s x i
        v = einops.rearrange(v, 'b (h c) s i -> b c h s i', h=self.num_heads)                               # b x c x h x s x i
        
        a = torch.softmax(math.sqrt(self.c)**-1 * torch.einsum('bchsi,bchti->bhsti', q, k), dim=-2)         # b x h x s x t x i
        a = einops.repeat(a, 'b h s t i -> b c h s t i', c=v.shape[1])                                      # b x c x h x s x t x i
        v = einops.repeat(v, 'b c h s t -> b c h s t i', i=a.shape[-1])                                     # b x c x h x s x t x i

        o = g * torch.einsum('bchsti,bchsti->bchsi', a, v)                                                  # b x c x h x s x i
        
        o = einops.rearrange(o, 'b c h s i -> b (c h) s i')                                                 # b x (c * h) x s x i
        msa_rep = self.w_rep(o)                                                                             # b x c_m x s x i

        return msa_rep
        

class Transition(nn.Module):
    def __init__(self, h, w, c, n=4):
        super().__init__()
        
        self.ln1 = nn.LayerNorm([c, h, w])
        self.l1 = nn.Conv2d(c, c * n, 1, bias=False)
        self.l2 = nn.Conv2d(c * n, c, 1, bias=False)
        
    def forward(self, msa_rep):
        msa_rep = self.ln1(msa_rep)
        a = self.l1(msa_rep)
        msa_rep = self.l2(F.relu(a))
        
        return msa_rep
        
        
class OPMean(nn.Module):
    def __init__(self, c_m, h_m, w_m, c_z, c):
        super().__init__()
        
        self.ln1 = nn.LayerNorm([c_m, h_m, w_m])
        self.w_a = nn.Conv2d(c_m, c, 1)
        self.w_b = nn.Conv2d(c_m, c, 1)
        self.w_z = nn.Conv2d(c * c, c_z, 1)
        
    def forward(self, msa_rep):
        msa_rep = self.ln1(msa_rep)                                                         # b x c_m x s x i
        a = self.w_a(msa_rep)                                                               # b x c x s x i
        b = self.w_b(msa_rep)                                                               # b x c x s x i
        o = torch.mean(torch.einsum('bcsi,bvsj->bcvsij', a, b), dim=3).flatten(1, 2)        # b x (c * c) x i x j
        z = self.w_z(o)                                                                     # b x c_m x s x i
        
        return z
        
        
class TriangleOutgoing(nn.Module):
    def __init__(self, c_z, h_z, w_z, c):
        super().__init__()
        
        self.ln1 = nn.LayerNorm([c_z, h_z, w_z])
        self.w_a_1 = nn.Conv2d(c_z, c, 1)
        self.w_a_2 = nn.Conv2d(c_z, c, 1)
        self.w_b_1 = nn.Conv2d(c_z, c, 1)
        self.w_b_2 = nn.Conv2d(c_z, c, 1)
        
        self.w_g = nn.Conv2d(c_z, c_z, 1)
        
        self.ln2 = nn.LayerNorm([c, h_z, w_z])
        self.w_rep = nn.Conv2d(c, c_z, 1)
        
        
    def forward(self, pair_rep):
        pair_rep = self.ln1(pair_rep)                                                   # b x c_z x i x j
        a = torch.sigmoid(self.w_a_1(pair_rep)) * self.w_a_2(pair_rep)                  # b x c x i x j
        b = torch.sigmoid(self.w_b_1(pair_rep)) * self.w_b_2(pair_rep)                  # b x c x i x j
        g = torch.sigmoid(self.w_g(pair_rep))                                           # b x c_z x i x j
        
        pair_rep = g * self.w_rep(self.ln2(torch.einsum('bcik,bcjk->bcij', a, b)))      # b x c_z x i x j
        
        return pair_rep
        
    
class TriangleIncoming(nn.Module):
    def __init__(self, c_z, h_z, w_z, c):
        super().__init__()
        
        self.ln1 = nn.LayerNorm([c_z, h_z, w_z])
        self.w_a_1 = nn.Conv2d(c_z, c, 1)
        self.w_a_2 = nn.Conv2d(c_z, c, 1)
        self.w_b_1 = nn.Conv2d(c_z, c, 1)
        self.w_b_2 = nn.Conv2d(c_z, c, 1)
        
        self.w_g = nn.Conv2d(c_z, c_z, 1)
        
        self.ln2 = nn.LayerNorm([c, h_z, w_z])
        self.w_rep = nn.Conv2d(c, c_z, 1)
        
        
    def forward(self, pair_rep):
        pair_rep = self.ln1(pair_rep)                                                   # b x c_z x i x j
        a = torch.sigmoid(self.w_a_1(pair_rep)) * self.w_a_2(pair_rep)                  # b x c x i x j
        b = torch.sigmoid(self.w_b_1(pair_rep)) * self.w_b_2(pair_rep)                  # b x c x i x j
        g = torch.sigmoid(self.w_g(pair_rep))                                           # b x c_z x i x j
        
        pair_rep = g * self.w_rep(self.ln2(torch.einsum('bcki,bckj->bcij', a, b)))         # b x c_z x i x j
        
        return pair_rep
        
        
class TriangleAttentionStart(nn.Module):
    def __init__(self, c_z, h_z, w_z, num_heads, c):
        super().__init__()
        
        self.c = c
        self.num_heads = num_heads
        
        self.ln1 = nn.LayerNorm([c_z, h_z, w_z])
        self.w_q = nn.Conv2d(c_z, c * num_heads, 1, bias=False)
        self.w_k = nn.Conv2d(c_z, c * num_heads, 1, bias=False)
        self.w_v = nn.Conv2d(c_z, c * num_heads, 1, bias=False)
        
        self.w_b = nn.Conv2d(c_z, num_heads, 1, bias=False)
        
        self.w_g = nn.Conv2d(c_z, c * num_heads, 1)
        
        self.w_rep = nn.Conv2d(c * num_heads, c_z, 1)
        
    def forward(self, pair_rep):
        # Input Projections
        pair_rep = self.ln1(pair_rep)
        q = self.w_q(pair_rep)                                                                                              # b x (c * h) x i x j
        k = self.w_k(pair_rep)                                                                                              # b x (c * h) x i x j
        v = self.w_v(pair_rep)                                                                                              # b x (c * h) x i x j
        
        b = self.w_b(pair_rep)                                                                                              # b x h x i x j
        
        g = torch.sigmoid(self.w_g(pair_rep))                                                                               # b x (c * h) x i x j
        g = einops.rearrange(g, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        
        # Attention
        q = einops.rearrange(q, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        k = einops.rearrange(k, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        v = einops.rearrange(v, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        
        a = torch.softmax(math.sqrt(self.c)**-1 * torch.einsum('bchij,bchik->bhijk', q, k) + b.unsqueeze(2), dim=-1)        # b x h x i x j x k
        
        a = einops.repeat(a, 'b h i j k -> b c h i j k', c=v.shape[1])                                                      # b x c x h x i x j x k
        v = einops.repeat(v, 'b c h i k -> b c h i j k', j=a.shape[-2])                                                     # b x c x h x i x j x k

        o = g * torch.einsum('bchijk,bchijk->bchij', a, v)                                                                  # b x c x h x i x j
        
        # Output 
        o = einops.rearrange(o, 'b c h i j -> b (c h) i j')                                                                 # b x (c * h) x i x j
        pair_rep = self.w_rep(o)                                                                                            # b x c_z x s x i

        return pair_rep
        
        
        
class TriangleAttentionEnd(nn.Module):
    def __init__(self, c_z, h_z, w_z, num_heads, c):
        super().__init__()
        
        self.c = c
        self.num_heads = num_heads
        
        self.ln1 = nn.LayerNorm([c_z, h_z, w_z])
        self.w_q = nn.Conv2d(c_z, c * num_heads, 1, bias=False)
        self.w_k = nn.Conv2d(c_z, c * num_heads, 1, bias=False)
        self.w_v = nn.Conv2d(c_z, c * num_heads, 1, bias=False)
        
        self.w_b = nn.Conv2d(c_z, num_heads, 1, bias=False)
        
        self.w_g = nn.Conv2d(c_z, c * num_heads, 1)
        
        self.w_rep = nn.Conv2d(c * num_heads, c_z, 1)
        
    def forward(self, pair_rep):
        # Input Projections
        pair_rep = self.ln1(pair_rep)
        q = self.w_q(pair_rep)                                                                                              # b x (c * h) x i x j
        k = self.w_k(pair_rep)                                                                                              # b x (c * h) x i x j
        v = self.w_v(pair_rep)                                                                                              # b x (c * h) x i x j
        
        b = self.w_b(pair_rep)                                                                                              # b x h x i x j
        
        g = torch.sigmoid(self.w_g(pair_rep))                                                                               # b x (c * h) x i x j
        g = einops.rearrange(g, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        
        # Attention
        q = einops.rearrange(q, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        k = einops.rearrange(k, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        v = einops.rearrange(v, 'b (h c) i j -> b c h i j', h=self.num_heads)                                               # b x c x h x i x j
        
        b = einops.rearrange(b, 'b h i j -> b h j 1 i')                                                                     # b x h x j x 1 x i
        a = torch.softmax(math.sqrt(self.c)**-1 * torch.einsum('bchij,bchkj->bhijk', q, k) + b, dim=-1)                     # b x h x i x j x k
        
        a = einops.repeat(a, 'b h i j k -> b c h i j k', c=v.shape[1])                                                      # b x c x h x i x j x k
        v = einops.repeat(v, 'b c h k j -> b c h i j k', i=a.shape[-3])                                                     # b x c x h x i x j x k

        o = g * torch.einsum('bchijk,bchijk->bchij', a, v)                                                                  # b x c x h x i x j
        
        # Output 
        o = einops.rearrange(o, 'b c h i j -> b (c h) i j')                                                                 # b x (c * h) x i x j
        pair_rep = self.w_rep(o)                                                                                            # b x c_z x s x i

        return pair_rep

        
class RowWiseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        assert len(x.shape) == 4
        if self.training:
            mask = torch.empty((x.shape[0], x.shape[1], x.shape[2], 1), device=x.device)
            mask.bernoulli_(p=self.p)
            x.mul_(mask) 
            
        return x
        

class ColWiseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            mask = torch.empty((x.shape[0], x.shape[1], 1, x.shape[3]), device=x.device)
            mask.bernoulli_(p=self.p)
            x.mul_(mask) 
            
        return x            
    
    
if __name__ == '__main__':
    num_blocks = 48
    c_m = 384
    h_m = 5
    w_m = 5
    a_m = 32
    heads_m = 8
    c_z = 128
    h_z = 5
    w_z = 5
    a_z = 128
    heads_z = 4
    c_s = 384
    
    evoformer = EvoFormer()
    evoformer.cuda()
    msa_rep = torch.randn([32, c_m, h_m, w_m], device='cuda', requires_grad=True)
    pair_rep = torch.randn([32, c_z, h_z, w_z], device='cuda', requires_grad=True)
    
    msa_rep, pair_rep, s = evoformer(msa_rep, pair_rep)
    loss = msa_rep.sum()
    
    loss.backward()
