import torch
import torch.nn as nn


class InputEmbedder(nn.Module):
    def __init__(self, c_m, c_z, n_clust):
        super().__init__()
        
        self.w_a = nn.Conv1d(20, c_z, 1)
        self.w_b = nn.Conv1d(20, c_z, 1)
        
        self.v_bins = torch.arange(-32, 33, dtype=int).view(-1, 1, 1)
        self.w_p = nn.Conv2d(65, c_z, 1)
        
        self.w_clust = nn.ModuleList([nn.Conv1d(21, c_m, 1) for _ in range(n_clust)])
        self.w_m = nn.Conv1d(20, c_m, 1)
    
    # Algorithm 3
    def forward(self, target_feat, residue_index, msa_feat):
        a = self.w_a(target_feat)
        b = self.w_b(target_feat)
        
        a = a.unsqueeze(-1).expand(a.shape[0], a.shape[1], a.shape[2], a.shape[2])
        b = b.unsqueeze(-1).expand(b.shape[0], b.shape[1], b.shape[2], b.shape[2])
        
        pair_rep = a + b.permute(0, 1, 3, 2)
        
        pair_rep += self._rel_pos(residue_index)
        
        msa_rep = []
        for w_clust in self.w_clust:
            msa_rep.append(w_clust(msa_feat))
        msa_rep = torch.stack(msa_rep, dim=-2)
        
        msa_rep += self.w_m(target_feat).unsqueeze(-2)
            
        return msa_rep, pair_rep

    # Algorithm 4
    def _rel_pos(self, residue_index):
        tmp = residue_index.unsqueeze(1).expand(residue_index.shape[0],
                                                residue_index.shape[1],
                                                residue_index.shape[1])
        d = tmp - tmp.permute(0, 2, 1)
        
        p = self.w_p(self._one_hot(d))
        
        return p
    
    # Algorithm 5
    def _one_hot(self, x):
        b = torch.argmin(torch.abs(x - self.v_bins), dim=1)

        return b
        
        