import torch.nn as nn

from evoformer import EvoFormer
from input_embedder import InputEmbedder


class Part1Model(nn.Module):
    def __init__(self, num_blocks, c_m, h_m, w_m, a_m, heads_m, c_z, h_z, w_z, a_z, heads_z, c_s):
        super().__init__()
        
        self.input_embedder = InputEmbedder(c_m=c_m, c_z=c_z, n_clust=h_m)
        
        self.evoformer = EvoFormer(num_blocks=num_blocks,
                                   c_m=c_m,
                                   h_m=h_m,
                                   w_m=w_m,
                                   a_m=a_m,
                                   heads_m=heads_m,
                                   c_z=c_z,
                                   h_z=h_z,
                                   w_z=w_z,
                                   a_z=a_z,
                                   heads_z=heads_z,
                                   c_s=c_s)
        
        self.dist_head = 0
        self.angle_head = 0
        
    def forward(self, msa_rep, pair_rep):
        msa_rep, pair_rep = self.evoformer(msa_rep, pair_rep)
        
        dist_pred = self.dist_head(...)
        angle_pred = self.angle_head(...)
        
        return dist_pred, angle_pred