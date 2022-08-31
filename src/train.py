import collections
import os

import sidechainnet as scn
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from apex.parallel import DistributedDataParallel as DDP
from sidechainnet.dataloaders.collate import pad_for_batch
from sidechainnet.utils.sequence import VOCAB, DSSPVocabulary
from tqdm import tqdm

from data import get_input
from model import Alphafold2

# Model Params
NUM_BLOCKS = 2  # 48
CROP_SIZE = 256  # 256
NUM_CLUST = 16
C_M = C_S = 256  # 384
A_M = 32
HEADS_M = 4  # 8
C_Z = 128  # 128
A_Z = 128  # 128
HEADS_Z = 2  # 4
NUM_LAYERS_STRUCTURE = 4
C_STRUCTURE = 128

# Train Params
EPOCHS = 21
CHECKPOINT_FREQUENCY = 2
LR = 1e-4

CHECKPOINT = None


# Need all this code to get DDP to work, all from scn.
Batch = collections.namedtuple(
    "Batch",
    "pids seqs msks evos secs angs "
    "crds int_seqs seq_evo_sec resolutions is_modified "
    "lengths str_seqs",
)


def collate_fn(insts):
    """Collates items extracted from a ProteinDataset, returning all items separately.
    Args:
        insts: A list of tuples, each containing one pnid, sequence, mask, pssm,
            angle, and coordinate extracted from a corresponding ProteinDataset.
        aggregate_input: A boolean that, if True, aggregates the 'model input'
            components of the data, or the data items (seqs, pssms) from which
            coordinates and angles are predicted.
    Returns:
        A tuple of the same information provided in the input, where each data type
        has been extracted in a list of its own. In other words, the returned tuple
        has one list for each of (pnids, seqs, msks, pssms, angs, crds). Each item in
        each list is padded to the maximum length of sequences in the batch.
    """
    # Instead of working with a list of tuples, we extract out each category of info
    # so it can be padded and re-provided to the user.
    (
        pnids,
        sequences,
        masks,
        pssms,
        secs,
        angles,
        coords,
        resolutions,
        mods,
        str_seqs,
    ) = list(zip(*insts))
    lengths = tuple(len(s) for s in sequences)
    max_batch_len = max(lengths)

    int_seqs = pad_for_batch(
        sequences, max_batch_len, "seq", seqs_as_onehot=False, vocab=VOCAB
    )
    padded_seqs = pad_for_batch(
        sequences, max_batch_len, "seq", seqs_as_onehot=True, vocab=VOCAB
    )
    padded_secs = pad_for_batch(
        secs, max_batch_len, "seq", seqs_as_onehot=True, vocab=DSSPVocabulary()
    )
    padded_msks = pad_for_batch(masks, max_batch_len, "msk")
    padded_pssms = pad_for_batch(pssms, max_batch_len, "pssm")
    padded_angs = pad_for_batch(angles, max_batch_len, "ang")
    padded_crds = pad_for_batch(coords, max_batch_len, "crd")
    padded_mods = pad_for_batch(mods, max_batch_len, "msk")

    # Non-aggregated model input
    return Batch(
        pids=pnids,
        seqs=padded_seqs,
        msks=padded_msks,
        evos=padded_pssms,
        secs=padded_secs,
        angs=padded_angs,
        crds=padded_crds,
        int_seqs=int_seqs,
        seq_evo_sec=None,
        resolutions=resolutions,
        is_modified=padded_mods,
        lengths=lengths,
        str_seqs=str_seqs,
    )


def main(rank, ngpus_per_node, world_size):
    if ngpus_per_node > 1:
        torch.cuda.set_device(rank)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        import torch.distributed as dist

        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

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
    model.cuda(rank)

    train_loader = scn.load(
        casp_version=7,
        with_pytorch="dataloaders",
        seq_as_onehot=True,
        aggregate_model_input=False,
        num_workers=8,
        dynamic_batching=False,
        batch_size=ngpus_per_node,
        complete_structures_only=True,
    )["train"]
    train_loader.collate_fn = collate_fn
    val_loader = scn.load(
        casp_version=7,
        with_pytorch="dataloaders",
        seq_as_onehot=True,
        aggregate_model_input=False,
        num_workers=8,
        dynamic_batching=False,
        batch_size=ngpus_per_node,
        complete_structures_only=True,
    )["valid-10"]
    val_loader.collate_fn = collate_fn

    optimizer = optim.Adam(model.parameters(), LR)

    if CHECKPOINT:
        checkpoint = torch.load(CHECKPOINT)
        checkpoint["model"] = {
            k.split("module.")[1]: v for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if ngpus_per_node > 1:
        model = DDP(model)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        if rank == 0:
            itr = tqdm(train_loader)
        else:
            itr = train_loader
        for batch in itr:
            seq, mask, evo, angle, t_r, t_t, x = get_input(CROP_SIZE, batch)

            if seq is None:
                continue

            if ngpus_per_node > 1:
                # Send each batch element to a different gpu.
                batch_size = len(seq)
                if rank < batch_size:
                    seq = seq[rank].unsqueeze(0).cuda(rank)
                    mask = mask[rank].unsqueeze(0).cuda(rank)
                    evo = evo[rank].unsqueeze(0).cuda(rank)
                    angle = angle[rank].unsqueeze(0).cuda(rank)
                    t_r = t_r[rank].unsqueeze(0).cuda(rank)
                    t_t = t_t[rank].unsqueeze(0).cuda(rank)
                    x = x[rank].unsqueeze(0).cuda(rank)
                else:
                    continue
            else:
                seq = seq.cuda()
                mask = mask.cuda()
                evo = evo.cuda()
                angle = angle.cuda()
                t_r = t_r.cuda()
                t_t = t_t.cuda()
                x = x.cuda()

            _, loss = model(seq, evo, (t_r, t_t), angle, x, mask)
            train_loss += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                seq, mask, evo, angle, t_r, t_t, x = get_input(CROP_SIZE, batch)

                if ngpus_per_node > 1:
                    batch_size = len(seq)
                    if rank < batch_size:
                        seq = seq[rank].unsqueeze(0).cuda(rank)
                        mask = mask[rank].unsqueeze(0).cuda(rank)
                        evo = evo[rank].unsqueeze(0).cuda(rank)
                        angle = angle[rank].unsqueeze(0).cuda(rank)
                        t_r = t_r[rank].unsqueeze(0).cuda(rank)
                        t_t = t_t[rank].unsqueeze(0).cuda(rank)
                        x = x[rank].unsqueeze(0).cuda(rank)
                    else:
                        continue
                else:
                    seq = seq.cuda()
                    mask = mask.cuda()
                    evo = evo.cuda()
                    angle = angle.cuda()
                    t_r = t_r.cuda()
                    t_t = t_t.cuda()
                    x = x.cuda()

                _, loss = model(seq, evo, (t_r, t_t), angle, x, mask)
                val_loss += loss

        train_loss = train_loss.item() / len(train_loader)
        val_loss = val_loss.item() / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch} Train Loss: {train_loss:.2f}")
            print(f"Epoch {epoch} Validation Loss: {val_loss:.2f}")

        if epoch % CHECKPOINT_FREQUENCY == 0 and rank == 0:
            print("Checkpointing...")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{epoch}-{val_loss:.2f}")


if __name__ == "__main__":
    ngpus_per_node = torch.cuda.device_count()

    if ngpus_per_node > 1:
        mp.spawn(main, args=(ngpus_per_node, ngpus_per_node), nprocs=ngpus_per_node)
    else:
        main(0, 1, 1)
