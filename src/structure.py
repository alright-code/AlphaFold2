import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureModule(nn.Module):
    def __init__(self, c_s, c_z, n_layer, c):
        super().__init__()

        self.ln_s_1 = nn.LayerNorm(c_s)
        self.ln_z = nn.LayerNorm(c_z)
        self.linear_s_1 = nn.Conv1d(c_s, c_s, 1)

        self.n_layer = n_layer

        self.invariant_point_attention = IPA(c_s, c_z)

        self.dropout = nn.Dropout(0.1)
        self.ln_s_2 = nn.LayerNorm(c_s)

        self.transition = nn.Sequential(
            nn.Conv1d(c_s, c_s, 1),
            nn.ReLU(),
            nn.Conv1d(c_s, c_s, 1),
            nn.ReLU(),
            nn.Conv1d(c_s, c_s, 1),
        )

        self.ln_s_3 = nn.LayerNorm(c_s)

        self.linear_a_1 = nn.Conv1d(c_s, c, 1)
        self.linear_a_2 = nn.Conv1d(c_s, c, 1)
        self.seqential_a_1 = nn.Sequential(
            nn.ReLU(), nn.Conv1d(c, c, 1), nn.ReLU(), nn.Conv1d(c, c, 1)
        )
        self.seqential_a_2 = nn.Sequential(
            nn.ReLU(), nn.Conv1d(c, c, 1), nn.ReLU(), nn.Conv1d(c, c, 1)
        )
        self.linear_a_3 = nn.Conv1d(c, 4, 1)

        self.backbone_update = BackboneUpdate(c_s)

    def forward(self, s_initial, z, t_true, alpha_true, x_true, mask):
        # s_initial - b x c_s x crop_size
        # z - b x c_z x crop_size x crop_size
        # t_true - b x crop_size x 3 x 3 and b x crop_size x 3
        # alpha_true - b x 2 x f x crop_size
        # x_true - b x crop_size x 3
        # mask - b x crop_size
        
        s_initial = self.ln_s_1(s_initial.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.ln_z(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        s = self.linear_s_1(s_initial)
        t_r = einops.repeat(torch.eye(3, dtype=torch.float), "i j -> b n i j", b=z.shape[0], n=z.shape[-1]).to(s.device).contiguous()
        t_t = torch.zeros([z.shape[0], z.shape[-1], 3], dtype=torch.float).to(s.device)
        t = (t_r, t_t)

        l_aux = torch.empty(self.n_layer)
        for l in range(1, self.n_layer + 1):  # shared weights
            s = s + self.invariant_point_attention(s, z, t)
            s = self.ln_s_2(self.dropout(s).permute(0, 2, 1)).permute(0, 2, 1)

            # Transition.
            s = s + self.transition(s)
            s = self.ln_s_3(self.dropout(s).permute(0, 2, 1)).permute(0, 2, 1)

            # Update Backbone.
            t = _compose(t, self.backbone_update(s))

            # Predict side chain torsion angles.
            a = self.linear_a_1(s) + self.linear_a_2(s_initial)
            a = a + self.seqential_a_1(a)
            a = a + self.seqential_a_2(a)
            alpha = self.linear_a_3(F.relu(a))
            alpha = einops.rearrange(alpha, "b (c d) n -> b c d n", c=2)
            
            # Auxiliary losses in every iteration.
            t_r, t_t = t
            x = t_t
            l_aux[l - 1] = self._compute_fape(
                t,
                x,
                t_true,
                x_true,
                mask,
                eps=1e-12
            ) + self._torsion_angle_loss(alpha, alpha_true, mask)

            # No rotation gradients between iterations to stabilize training.
            if l < self.n_layer:
                t = (t_r.detach(), t_t)
            else:
                t = (t_r, t_t)

        l_aux = l_aux.mean()
        
        return x, l_aux

    def _compute_fape(self, t, x, tt, xt, mask, eps, z=10):
        d_clamp = torch.tensor(10.).to(x.device)
        
        t_r_inv, t_t_inv = _inverse(t)
        x = torch.einsum("bixy,bjy->bijx", t_r_inv, x) + einops.repeat(
            t_t_inv, "b n c -> b n m c", m=x.shape[1]
        )

        tt_r_inv, tt_t_inv = _inverse(tt)
        xt = torch.einsum("bixy,bjy->bijx", tt_r_inv, xt) + einops.repeat(
            tt_t_inv, "b n c -> b n m c", m=x.shape[1]
        )

        mask = torch.einsum("bi,bj->bij", mask, mask)

        d = torch.sqrt(torch.norm(x - xt, dim=-1) ** 2 + eps)

        loss = (1 / z) * torch.mean(torch.min(d_clamp, d)[torch.where(mask)])
        
        return loss

    def _torsion_angle_loss(self, alpha, alpha_true, mask):
        # b x f x 2 x n
        l = torch.norm(alpha, dim=2)
        alpha_pred = alpha / l

        loss_torsion = torch.sum(
            torch.norm(alpha_pred - alpha_true, dim=2) ** 2 * mask
        ) / max(mask.sum() * 2, 1)
        
        loss_anglenorm = torch.sum(torch.abs(l - 1) * mask) / max(mask.sum() * 2, 1)
        
        return loss_torsion + 0.02 * loss_anglenorm


class IPA(nn.Module):
    def __init__(self, c_s, c_z, n_heads=12, c=16, n_query_points=4, n_point_values=8):
        super().__init__()

        self.weights_q = nn.Conv1d(c_s, c * n_heads, 1, bias=False)
        self.weights_k = nn.Conv1d(c_s, c * n_heads, 1, bias=False)
        self.weights_v = nn.Conv1d(c_s, c * n_heads, 1, bias=False)

        self.weights_qp = nn.Conv1d(c_s, 3 * n_heads * n_query_points, 1, bias=False)
        self.weights_kp = nn.Conv1d(c_s, 3 * n_heads * n_query_points, 1, bias=False)
        self.weights_vp = nn.Conv1d(c_s, 3 * n_heads * n_point_values, 1, bias=False)

        self.weights_b = nn.Conv2d(c_z, n_heads, 1, bias=False)

        self.w_c = math.sqrt(2 / (9 * n_query_points))
        self.w_l = math.sqrt(1 / 3)
        self.c = c

        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values

        self.gamma = nn.Parameter(torch.empty([n_heads, 1, 1]))
        torch.nn.init.xavier_uniform_(self.gamma)

        self.weights_s = nn.Conv1d(c_z * n_heads + c*n_heads + 3*n_heads*n_point_values + 1, c_s, 1)

    def forward(self, s, z, t):
        q = self.weights_q(s)
        k = self.weights_k(s)
        v = self.weights_v(s)
        qp = self.weights_qp(s)
        kp = self.weights_kp(s)
        vp = self.weights_vp(s)

        q = einops.rearrange(q, "b (c h) i -> b c h i", h=self.n_heads)
        k = einops.rearrange(k, "b (c h) i -> b c h i", h=self.n_heads)
        v = einops.rearrange(v, "b (c h) i -> b c h i", h=self.n_heads)
        qp = einops.rearrange(
            qp, "b (c h p) i -> b c h p i", h=self.n_heads, p=self.n_query_points
        )
        kp = einops.rearrange(
            kp, "b (c h p) i -> b c h p i", h=self.n_heads, p=self.n_query_points
        )
        vp = einops.rearrange(
            vp, "b (c h p) i -> b c h p i", h=self.n_heads, p=self.n_point_values
        )

        # t    b x n x 3 x 3
        # qp   b x 3 x heads x p x n
        tmp1 = _apply(t, qp)  # b x 3 x heads x p x n
        tmp1 = einops.repeat(
            tmp1, "b c h p n -> b c h p n m", m=tmp1.shape[-1]
        )  # b x 3 x h x p x n x n
        tmp2 = _apply(t, kp)  # b x 3 x heads x p x n
        tmp2 = einops.repeat(
            tmp2, "b c h p n -> b c h p m n", m=tmp2.shape[-1]
        )  # b x 3 x h x p x n x n

        tmp = (F.softplus(self.gamma) * self.w_c / 2) * torch.sum(
            torch.norm(tmp1 - tmp2, dim=1) ** 2, dim=-3
        )  # b x h x n x n

        b = self.weights_b(z)
        a = torch.softmax(
            self.w_l
            * (
                (
                    1 / math.sqrt(self.c) * torch.einsum("bchi,bchj->bhij", q, k)
                    + b
                    - tmp
                )
            ), dim=-1
        )

        o1 = torch.einsum("bhij,bcij->bchi", a, z)
        o1 = einops.rearrange(o1, "b c h i -> b (c h) i")
        o2 = torch.einsum("bhij,bchj->bchi", a, v)
        o2 = einops.rearrange(o2, "b c h i -> b (c h) i")
        o3_ = torch.einsum("bhij,bchpj->bchpi", a, _apply(t, vp))
        o3 = _apply(_inverse(t), o3_)
        o3 = einops.rearrange(o3, "b c h p i -> b (c h p) i")

        # c_z*h + c*h + 3*h*n_point_values + 1
        s = self.weights_s(
            torch.cat(
                [
                    o1,
                    o2,
                    o3,
                    torch.norm(o3, dim=1).unsqueeze(1),
                ],
                dim=1,
            )
        )

        return s


def _apply(t, x):
    t_r, t_t = t

    x_global = torch.einsum("bnij,bjhpn->bihpn", t_r, x) + einops.rearrange(
        t_t, "b n a -> b a 1 1 n"
    )

    return x_global


def _inverse(t):
    t_r, t_t = t

    t_r = t_r.transpose(-1, -2)
    t_t = torch.einsum("bnij,bnj->bni", -t_r, t_t)

    return t_r, t_t


def _compose(t1, t2):
    t_r_1, t_t_1 = t1
    t_r_2, t_t_2 = t2
    
    t_r = torch.einsum("bnij,bnjk->bnik", t_r_1, t_r_2)
    t_t = torch.einsum("bnij,bnj->bni", t_r_1, t_t_2) + t_t_1
    
    return t_r, t_t


class BackboneUpdate(nn.Module):
    def __init__(self, c_s):
        super().__init__()

        self.linear = nn.Conv1d(c_s, 6, 1)

    def forward(self, s):
        out = self.linear(s)
        a = 1
        b = out[:, 0, :]
        c = out[:, 1, :]
        d = out[:, 2, :]
        t_t = out[:, 3:, :]
        
        t_t = einops.rearrange(t_t, "b i n -> b n i")

        denom = torch.sqrt(1 + b**2 + c**2 + d**2)
        a = a / denom
        b = b / denom
        c = c / denom
        d = d / denom
        
        idx00 = (a**2 + b**2 - c**2 - d**2).unsqueeze(1)
        idx01 = (2 * b * c - 2 * a * d).unsqueeze(1)
        idx02 = (2 * b * d + 2 * a * c).unsqueeze(1)
        row0 = torch.cat([idx00, idx01, idx02], dim=1).unsqueeze(1)
        idx10 = (2 * b * c + 2 * a * d).unsqueeze(1)
        idx11 = (a**2 - b**2 + c**2 - d**2).unsqueeze(1)
        idx12 = (2 * c * d - 2 * a * b).unsqueeze(1)
        row1 = torch.cat([idx10, idx11, idx12], dim=1).unsqueeze(1)
        idx20 = (2 * b * d - 2 * a * c).unsqueeze(1)
        idx21 = (2 * c * d + 2 * a * b).unsqueeze(1)
        idx22 = (a**2 - b**2 - c**2 + d**2).unsqueeze(1)
        row2 = torch.cat([idx20, idx21, idx22], dim=1).unsqueeze(1)
        
        t_r = torch.cat([row0, row1, row2], dim=1)
        t_r = einops.rearrange(t_r, "b i j n -> b n i j")

        # b x n x 3 x 3
        return t_r, t_t


if __name__ == '__main__':
    crop_size = 256
    c_s = 64
    c_z = 64
    n_layer = 1
    c = 16
    
    model = StructureModule(crop_size, c_s, c_z, n_layer, c)
    
    s_initial = torch.randn([1, c_s, crop_size])
    z = torch.randn([1, c_z, crop_size, crop_size])
    t_true = (torch.randn([1, crop_size, 3, 3]), torch.randn([1, crop_size, 3]))
    alpha_true = 0
    x_true = torch.randn([1, crop_size, 3])
    
    model(s_initial, z, t_true, alpha_true, x_true)
    
    