import einops
import sidechainnet as scn
import torch
import torch.nn.functional as F


def _rigid_from_three_points(x1, x2, x3):
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = F.normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.einsum("bnp,bnp->bn", e1, v2).unsqueeze(-1)
    e2 = F.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)

    r = torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)
    t = x2

    return r, t


def get_seq_features(batch):
    coords = batch.crds

    x1_idxs = torch.arange(0, coords.shape[1], 14)
    x2_idxs = torch.arange(1, coords.shape[1], 14)
    x3_idxs = torch.arange(2, coords.shape[1], 14)

    x1 = coords[:, x1_idxs, :]
    x2 = coords[:, x2_idxs, :]
    x3 = coords[:, x3_idxs, :]

    t_r, t_t = _rigid_from_three_points(x1, x2, x3)

    seqs = batch.seqs
    masks = batch.msks
    evos = batch.evos
    angles = batch.angs[:, :, 0:2]

    # Replace 0 padding with identity padding.
    t_r[torch.where(masks == 0)] = torch.eye(3)

    angs_x = torch.cos(angles).unsqueeze(-1)
    angs_y = torch.sin(angles).unsqueeze(-1)

    angs = torch.cat([angs_x, angs_y], dim=-1)

    # Remove data where we do not have information.
    keep = torch.ones(len(t_r), dtype=bool)
    for i in range(len(t_r)):
        try:
            torch.inverse(t_r[i])
        except:
            keep[i] = False

    seqs = seqs[keep]
    masks = masks[keep]
    evos = evos[keep]
    angs = angs[keep]
    t_r = t_r[keep]
    t_t = t_t[keep]
    x2 = x2[keep]

    return seqs, masks, evos, angs, t_r, t_t, x2


# Follow the crop start method in the paper.
def _get_crop_start(crop_size, length, clamped=True):
    n = length - crop_size
    if not clamped:
        x = torch.randint(n, size=(1,))
    else:
        x = 0

    pos = torch.randint(low=1, high=(n - x + 1), size=(1,))

    return pos


def get_input(crop_size, batch):
    seqs, masks, evos, angles, t_r, t_t, x = get_seq_features(batch)

    if len(seqs) == 0:
        return None, None, None, None, None, None, None

    # Handle sequences smaller than the crop size.
    pad_amount = max((crop_size - len(seqs[0])) // 2 + 1, 0)
    seqs = F.pad(seqs.permute(0, 2, 1), (pad_amount, pad_amount))
    masks = F.pad(masks, (pad_amount, pad_amount))
    evos = F.pad(evos.permute(0, 2, 1), (pad_amount, pad_amount))
    angles = F.pad(angles.permute(0, 2, 3, 1), (pad_amount, pad_amount))
    t_t = F.pad(t_t.permute(0, 2, 1), (pad_amount, pad_amount))
    x = F.pad(x.permute(0, 2, 1), (pad_amount, pad_amount))

    # Pad rotation with identity matrix.
    l_pad = einops.repeat(torch.eye(3), "i j -> b i j n", b=x.shape[0], n=pad_amount)
    r_pad = einops.repeat(torch.eye(3), "i j -> b i j n", b=x.shape[0], n=pad_amount)
    t_r = torch.cat([l_pad, t_r.permute(0, 2, 3, 1), r_pad], dim=-1)

    # Crop.
    crop_start = _get_crop_start(crop_size, seqs.shape[-1])
    seqs = seqs[:, :, crop_start : (crop_start + crop_size)]
    masks = masks[:, crop_start : (crop_start + crop_size)]
    evos = evos[:, :, crop_start : (crop_start + crop_size)]
    angles = angles[:, :, :, crop_start : (crop_start + crop_size)]
    t_r = t_r[:, :, :, crop_start : (crop_start + crop_size)]
    t_t = t_t[:, :, crop_start : (crop_start + crop_size)]
    x = x[:, :, crop_start : (crop_start + crop_size)]

    t_r = einops.rearrange(t_r, "b i j n -> b n i j")
    t_t = einops.rearrange(t_t, "b i n -> b n i")
    x = einops.rearrange(x, "b i n -> b n i")

    return seqs.float(), masks, evos.float(), angles, t_r, t_t, x


def get_input_test(batch):
    seqs, masks, evos, angles, t_r, t_t, x = get_seq_features(batch)

    # Handle sequences smaller than the crop size.
    seqs = seqs.permute(0, 2, 1)
    evos = evos.permute(0, 2, 1)
    angles = angles.permute(0, 2, 3, 1)

    return seqs.float(), masks, evos.float(), angles, t_r, t_t, x


if __name__ == "__main__":
    train_loader = scn.load(
        casp_version=7,
        with_pytorch="dataloaders",
        seq_as_onehot=True,
        aggregate_model_input=False,
        num_workers=8,
        dynamic_batching=False,
        batch_size=2,
    )["train"]

    for batch in train_loader:
        get_input(256, batch)
