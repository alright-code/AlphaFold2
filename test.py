import sidechainnet as scn
import torch
import matplotlib.pyplot as plt

from data import get_input_test
from model import Alphafold2
from train import *

BATCH_SIZE = 1

CHECKPOINT = "10-1.93"


def main():
    model = Alphafold2(
        num_blocks=NUM_BLOCKS,
        num_clust=NUM_CLUST,
        c_m=C_M,
        a_m=A_M,
        heads_m=HEADS_M,
        c_z=C_Z,
        a_z=A_Z,
        heads_z=HEADS_Z,
        c_s=C_S,
        num_layers_structure=NUM_LAYERS_STRUCTURE,
        c_structure=C_STRUCTURE,
    )
    model.cuda()

    test_loader = scn.load(
        casp_version=7,
        with_pytorch="dataloaders",
        seq_as_onehot=True,
        aggregate_model_input=False,
        num_workers=8,
        dynamic_batching=False,
        batch_size=BATCH_SIZE,
        complete_structures_only=True,
    )["test"]

    if CHECKPOINT:
        checkpoint = torch.load(CHECKPOINT)
        checkpoint["model"] = {
            k.split("module.")[1]: v for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(checkpoint["model"])

    model.eval()
    with torch.no_grad():
        losses = torch.empty(0)
        for i, batch in enumerate(test_loader):
            seq, mask, evo, angle, t_r, t_t, x = get_input_test(batch)
            seq = seq.cuda()
            mask = mask.cuda()
            evo = evo.cuda()
            angle = angle.cuda()
            t_r = t_r.cuda()
            t_t = t_t.cuda()
            x = x.cuda()

            x_pred, loss = model(seq, evo, (t_r, t_t), angle, x, mask)
            print(f"{loss.item():.2f}")

            a_pred = x_pred[torch.where(mask)][:, 0].cpu()
            b_pred = x_pred[torch.where(mask)][:, 1].cpu()
            c_pred = x_pred[torch.where(mask)][:, 2].cpu()

            a_true = x[torch.where(mask)][:, 0].cpu()
            b_true = x[torch.where(mask)][:, 1].cpu()
            c_true = x[torch.where(mask)][:, 2].cpu()

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.plot(a_pred, b_pred, c_pred, color="red")
            ax.plot(a_true, b_true, c_true, color="green")

            plt.savefig(f"{loss:.2f}.png")
            plt.close()

            losses = torch.cat([losses, loss.unsqueeze(0)])

    print(f"Test loss: {losses.mean().item():.2f}")


if __name__ == "__main__":
    main()
