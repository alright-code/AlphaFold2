import torch
import torch.nn as nn

from evoformer import EvoFormer
from input_embedder import InputEmbedder
from structure import StructureModule


class Alphafold2(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_clust,
        c_m,
        a_m,
        heads_m,
        c_z,
        a_z,
        heads_z,
        c_s,
        num_layers_structure,
        c_structure,
    ):
        super().__init__()

        self.input_embedder = InputEmbedder(c_m=c_m, c_z=c_z, num_clust=num_clust)

        self.evoformer = EvoFormer(
            num_blocks=num_blocks,
            c_m=c_m,
            a_m=a_m,
            heads_m=heads_m,
            c_z=c_z,
            a_z=a_z,
            heads_z=heads_z,
            c_s=c_s,
        )

        self.structure = StructureModule(
            c_s=c_s,
            c_z=c_z,
            n_layer=num_layers_structure,
            c=c_structure,
        )

    def forward(self, seq, evo, t_true, alpha_true, x_true, mask):
        residue_index = (
            torch.arange(seq.shape[-1], dtype=int)
            .expand(seq.shape[0], -1)
            .to(seq.device)
        )
        msa_rep, pair_rep = self.input_embedder(seq, residue_index, evo)

        msa_rep, pair_rep, s = self.evoformer(msa_rep, pair_rep)

        x, loss_aux = self.structure(
            s_initial=s,
            z=pair_rep,
            t_true=t_true,
            alpha_true=alpha_true,
            x_true=x_true,
            mask=mask,
        )

        return x, loss_aux
