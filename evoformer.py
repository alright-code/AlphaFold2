import einops
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvoFormer(nn.Module):
    def __init__(self, num_blocks=48):
        super().__init__()
        
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        
    def forward(self, msa_rep, pair_rep):
        return self.blocks(msa_rep, pair_rep)

        
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.a1 = RowWiseGatedAttention()
        self.a2 = ColWiseGatedAttention()
        self.trans1 = Transition()
        self.op_mean = OPMean()
        
        self.tri1 = TriangleOutgoing()
        self.tri2 = TriangleIncoming()
        self.tri3 = TriangleAttentionStart()
        self.tri4 = TriangleAttentionEnd()
        self.trans2 = Transition()
        
    def forward(self, msa_rep, pair_rep):
        msa_rep = msa_rep + self.a1(msa_rep, pair_rep)
        msa_rep = msa_rep + self.a2(msa_rep)
        msa_rep = msa_rep + self.trans1(msa_rep)
        
        pair_rep = pair_rep + self.op_mean(msa_rep)
        pair_rep = pair_rep + self.tri1(pair_rep)
        pair_rep = pair_rep + self.tri2(pair_rep)
        pair_rep = pair_rep + self.tri3(pair_rep)
        pair_rep = pair_rep + self.tri4(pair_rep)
        pair_rep = pair_rep + self.trans2(pair_rep)
        
        return msa_rep, pair_rep


class RowWiseGatedAttention(nn.Module):
    def __init__(self, h_m, w_m, h_z, w_z, c_m=256, c_z=128, c=32, num_heads=8):
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
    def __init__(self, h_m, w_m, c_m=256, c=32, num_heads=8):
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
    def __init__(self, h_m, w_m, c_m=256, n=4):
        super().__init__()
        
        self.ln1 = nn.LayerNorm([c_m, h_m, w_m])
        self.l1 = nn.Conv2d(c_m, c_m * n, 1, bias=False)
        self.l2 = nn.Conv2d(c_m * n, c_m, 1, bias=False)
        
    def forward(self, msa_rep):
        msa_rep = self.ln1(msa_rep)
        a = self.l1(msa_rep)
        msa_rep = self.l2(F.Relu(a))
        
        return msa_rep
        
        
class OPMean(nn.Module):
    def __init__(self, h_m, w_m, c_m=256, c_z=128, c=32):
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
    def __init__(self, h_z, w_z, c_z=128, c=128):
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
    def __init__(self, h_z, w_z, c_z=128, c=128):
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
    def __init__(self, h_z, w_z, c_z=128, c=32, num_heads=4):
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
    def __init__(self, h_z, w_z, c_z=128, c=32, num_heads=4):
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
    def __init__(self):
        super().__init__()
        
    def forward(self):
        ...
        
# GOOD   
def test1():
    I = J = 5
    S = 7
    C = 8
    H = 9
    B = 10
    
    q = torch.rand([B, C, H, S, I])
    k = torch.rand([B, C, H, S, J])
    bi = torch.rand([B, H, I, J])
    
    a1 = torch.empty([B, H, S, I, J])
    for b in range(B):    
        for h in range(H):
            for s in range(S):
                for i in range(I):
                    for j in range(J):
                        tmp = torch.dot(q[b, :, h, s, i], k[b, :, h, s, j])
                        a1[b, h, s, i, j] = math.sqrt(C)**-1 * tmp + bi[b, h, i, j]
    a1 = torch.softmax(a1, dim=-1)

    a2 = torch.softmax(math.sqrt(C)**-1 * torch.einsum('bchsi,bchsj->bhsij', q, k) + bi.unsqueeze(2), dim=-1)           # b x h x s x i x j
                    
    assert (a1 == a2).all()
    

# GOOD
def test2():
    I = J = 6
    S = 7
    C = 8
    H = 9
    B = 10
    
    a = torch.rand([B, H, S, I, J])
    g = torch.rand([B, C, H, S, I])
    v = torch.rand([B, C, H, S, I])
    
    o1 = torch.empty([B, C, H, S, I])
    for b in range(B):
        for h in range(H):
            for s in range(S):
                for i in range(I):
                    tmp = 0
                    for j in range(J):
                        tmp += a[b, h, s, i, j] * v[b, :, h, s, i]
                    tmp *= g[b, :, h, s, i]
                    o1[b, :, h, s, i] = tmp
    
    a_ = einops.repeat(a, 'b h s i j -> b c h s i j', c=v.shape[1])                                                      # b x c x h x s x i x j
    v_ = einops.repeat(v, 'b c h s i -> b c h s i j', j=a.shape[-1])                                                     # b x c x h x s x i x j

    o2 = g * torch.einsum('bchsij,bchsij->bchsi', a_, v_)                                                                # b x c x h x s x i
    
    assert (o1 == o2).all()
    

# GOOD
def test3():
    I = T = 6
    S = 7
    C = 8
    H = 9
    B = 10
    
    q = torch.rand([B, C, H, S, I])
    k = torch.rand([B, C, H, T, I])
    
    a1 = torch.empty([B, H, S, T, I])
    for b in range(B):
        for h in range(H):
            for s in range(S):
                for t in range(T):
                    for i in range(I):
                        tmp = torch.dot(q[b, :, h, s, i], k[b, :, h, t, i])
                        a1[b, h, s, t, i] = math.sqrt(C)**-1 * tmp
    a1 = torch.softmax(a1, dim=-2)
    
    a2 = torch.softmax(math.sqrt(C)**-1 * torch.einsum('bchsi,bchti->bhsti', q, k), dim=-2)            # b x h x s x t x i
    
    assert (a1 == a2).all()
    

# GOOD
def test4():
    I = T = 6
    S = 7
    C = 8
    H = 9
    B = 10
    
    g = torch.rand([B, C, H, S, I])
    a = torch.rand([B, H, S, T, I])
    v = torch.rand([B, C, H, S, T])
    
    o1 = torch.empty([B, C, H, S, I])
    for b in range(B):
        for h in range(H):
            for s in range(S):
                for i in range(I):
                    tmp = 0
                    for t in range(T):
                        tmp += a[b, h, s, t, i] * v[b, :, h, s, t]
                    tmp *= g[b, :, h, s, i]
                    o1[b, :, h, s, i] = tmp
    
    a_ = einops.repeat(a, 'b h s t i -> b c h s t i', c=v.shape[1])                                      # b x c x h x s x t x i
    v_ = einops.repeat(v, 'b c h s t -> b c h s t i', i=a.shape[-1])                                     # b x c x h x s x t x i

    o2 = g * torch.einsum('bchsti,bchsti->bchsi', a_, v_)                                                  # b x c x h x s x i
    
    assert (o1 == o2).all()
        

# GOOD
def test5():
    I = J = S = 6
    S = 7
    C = 8
    B = 10
    
    a = torch.rand([B, C, S, I]) * 10
    bi = torch.rand([B, C, S, I]) * 20
    
    o1 = torch.empty([B, C, C, I, J])
    for b in range(B):
        for i in range(I):
            for j in range(J):
                tmp = 0
                for s in range(S):
                    tmp += torch.einsum('i,j->ij', a[b, :, s, i], bi[b, :, s, j])
                tmp = tmp / S
                o1[b, :, :, i, j] = tmp
    o1 = o1.flatten(1, 2)        
            
    o2 = torch.mean(torch.einsum('bcsi,bvsj->bcvsij', a, bi), dim=3).flatten(1, 2)
    
    assert torch.abs(o1 - o2).max() < 0.0001
      

# GOOD
def test6():
    I = J = K = 7
    C = 10
    B = 11
    
    a = torch.rand([B, C, I, J])
    bi = torch.rand([B, C, I, J])
    g = torch.rand([B, C, I, J])
    
    z1 = torch.empty([B, C, I, J])
    for b in range(B):
        for i in range(I):
            for j in range(J):
                tmp = 0
                for k in range(K):
                    tmp += a[b, :, i, k] * bi[b, :, j, k]
                z1[b, :, i, j] = g[b, :, i, j] * tmp
            
    z2 = g * torch.einsum('bcik,bcjk->bcij', a, bi)
    
    assert (z1 == z2).all()
      

# GOOD
def test7():
    I = J = K = 7
    C = 10
    B = 10
    
    a = torch.rand([B, C, I, J])
    bi = torch.rand([B, C, I, J])
    g = torch.rand([B, C, I, J])
    
    z1 = torch.empty([B, C, I, J])
    for b in range(B):
        for i in range(I):
            for j in range(J):
                tmp = 0
                for k in range(K):
                    tmp += a[b, :, k, i] * bi[b, :, k, j]
                z1[b, :, i, j] = g[b, :, i, j] * tmp
            
    z2 = g * torch.einsum('bcki,bckj->bcij', a, bi)
    
    assert (z1 == z2).all()


# GOOD
def test8():
    I = J = K = 5
    C = 8
    H = 9
    B = 10
    
    q = torch.rand([B, C, H, I, J])
    ky = torch.rand([B, C, H, I, J])
    bi = torch.rand([B, H, I, J])
    
    a1 = torch.empty([B, H, I, J, K])  
    for b in range(B):  
        for h in range(H):
            for i in range(I):
                for j in range(J):
                    for k in range(K):
                        tmp = torch.dot(q[b, :, h, i, j], ky[b, :, h, i, k])
                        a1[b, h, i, j, k] = math.sqrt(C)**-1 * tmp + bi[b, h, j, k]
    a1 = torch.softmax(a1, dim=-1)

    a2 = torch.softmax(math.sqrt(C)**-1 * torch.einsum('bchij,bchik->bhijk', q, ky) + bi.unsqueeze(2), dim=-1)
                    
    assert (a1 == a2).all()
    
    
# GOOD
def test9():
    I = J = K = 5
    C = 8
    H = 9
    B = 12
    
    g = torch.rand([B, C, H, I, J])
    a = torch.rand([B, H, I, J, K])
    v = torch.rand([B, C, H, I, K])
    
    o1 = torch.empty([B, C, H, I, J])
    for b in range(B):
        for h in range(H):
            for i in range(I):
                for j in range(J):
                    tmp = 0
                    for k in range(K):
                        tmp += a[b, h, i, j, k] * v[b, :, h, i, k]
                    tmp *= g[b, :, h, i, j]
                    o1[b, :, h, i, j] = tmp

    a_ = einops.repeat(a, 'b h i j k -> b c h i j k', c=v.shape[1])                                                      # b x c x h x i x j x k
    v_ = einops.repeat(v, 'b c h i k -> b c h i j k', j=a.shape[-2])                                                     # b x c x h x i x j x k

    o2 = g * torch.einsum('bchijk,bchijk->bchij', a_, v_)   

    assert (o1 == o2).all()


# GOOD
def test10():
    I = J = K = 5
    C = 8
    H = 9
    B = 10
    
    q = torch.rand([B, C, H, I, J])
    ky = torch.rand([B, C, H, I, J])
    bi = torch.rand([B, H, I, J])
    
    a1 = torch.empty([B, H, I, J, K])
    for b in range(B):
        for h in range(H):
            for i in range(I):
                for j in range(J):
                    for k in range(K):
                        tmp = torch.dot(q[b, :, h, i, j], ky[b, :, h, k, j])
                        a1[b, h, i, j, k] = math.sqrt(C)**-1 * tmp + bi[b, h, k, i]
    a1 = torch.softmax(a1, dim=-1)

    # h i j -> h k i
    bi = einops.rearrange(bi, 'b h i j -> b h j 1 i')                                                             # b x h x j x 1 x i
    a2 = torch.softmax(math.sqrt(C)**-1 * torch.einsum('bchij,bchkj->bhijk', q, ky) + bi, dim=-1)
                    
    assert (a1 == a2).all()


# GOOD
def test11():
    I = J = K = 5
    C = 8
    H = 9
    B = 12
    
    g = torch.rand([B, C, H, I, J])
    a = torch.rand([B, H, I, J, K])
    v = torch.rand([B, C, H, I, K])
    
    o1 = torch.empty([B, C, H, I, J])
    for b in range(B):   
        for h in range(H):
            for i in range(I):
                for j in range(J):
                    tmp = 0
                    for k in range(K):
                        tmp += a[b, h, i, j, k] * v[b, :, h, k, j]
                    tmp *= g[b, :, h, i, j]
                    o1[b, :, h, i, j] = tmp

    a_ = einops.repeat(a, 'b h i j k -> b c h i j k', c=v.shape[1])                                                      # b x c x h x i x j x k
    v_ = einops.repeat(v, 'b c h k j -> b c h i j k', i=a.shape[-3])                                                     # b x c x h x i x j x k

    o2 = g * torch.einsum('bchijk,bchijk->bchij', a_, v_)

    assert (o1 == o2).all()    
    
    
if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()
    test10()
    test11()