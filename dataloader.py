import numpy as np
import sidechainnet as scn
import torch
import torch.nn.functional as F

from tqdm import tqdm


def get_seq_features(batch):
    '''
    Take a batch of sequence info and return the sequence (one-hot),
    evolutionary info and (phi, psi, omega) angles per position, 
    as well as position mask.
    Also return the distance matrix, and distance mask.
    '''
    str_seqs = batch.str_seqs # seq in str format
    seqs = batch.seqs # seq in one-hot format
    int_seqs = batch.int_seqs # seq in int format
    masks = batch.msks # which positions are valid
    lengths = batch.lengths # seq length
    evos = batch.evos # PSSM / evolutionary info
    angs = batch.angs[:,:,0:2] # torsion angles: phi, psi
    
    # use coords to create distance matrix from c-beta
    # except use c-alpha for G
    # coords[:, 4, :] is c-beta, and coords[:, 1, :] is c-alpha
    coords = batch.crds # seq coord info (all-atom)
    batch_xyz = []
    for i in range(coords.shape[0]):
        xyz = []
        xyz = [coords[i][cpos+4,:] 
               if masks[i][cpos//14] and str_seqs[i][cpos//14] != 'G'
               else coords[i][cpos+1,:]
               for cpos in range(0, coords[i].shape[0]-1, 14)]
        batch_xyz.append(torch.stack(xyz))
    batch_xyz = torch.stack(batch_xyz)
    # now create pairwise distance matrix
    dmats = torch.cdist(batch_xyz, batch_xyz)
    # create matrix mask (0 means i,j invalid)
    dmat_masks = torch.einsum('bi,bj->bij', masks, masks)
    
    return seqs.float(), evos.float(), angs, masks, dmats, dmat_masks


class Part1Dataloader():
    def __init__(self, mode, batch_size, crop_size):
        if mode == 'val':
            mode = 'valid-10'
            
        self.crop_size = crop_size
            
        self.dataloader = scn.load(casp_version=7, with_pytorch='dataloaders', 
                                   seq_as_onehot=True, aggregate_model_input=False,
                                   num_workers=8, dynamic_batching=True, batch_size=batch_size)[mode]
        
    def _get_crop_start(self, length, clamped=True):
        n = length - self.crop_size
        if not clamped:
            x = torch.randint(n, size=(1,))
        else:
            x = 0

        pos = torch.randint(low=1, high=(n - x + 1), size=(1,))
       
        return pos
    
    def __iter__(self):
        for batch in self.dataloader:
            seqs, evos, angles, _, dists, dist_masks = get_seq_features(batch)
            
            # Handle sequences smaller than the crop size.
            pad_amount = max((self.crop_size - len(seqs[0])) // 2 + 1, 0)
            seqs = F.pad(seqs.permute(0, 2, 1), (pad_amount, pad_amount))
            evos = F.pad(evos.permute(0, 2, 1), (pad_amount, pad_amount))
            angles = F.pad(angles.permute(0, 2, 1), (pad_amount, pad_amount))
            dists = F.pad(dists, (pad_amount, pad_amount, pad_amount, pad_amount))
            dist_masks = F.pad(dist_masks, (pad_amount, pad_amount, pad_amount, pad_amount))
            
            crop_start = self._get_crop_start(seqs.shape[-1])
            seqs = seqs[:, :, crop_start:(crop_start + self.crop_size)]
            evos = evos[:, :, crop_start:(crop_start + self.crop_size)]
            angles = angles[:, :, crop_start:(crop_start + self.crop_size)]
            dists = dists[:, crop_start:(crop_start + self.crop_size), crop_start:(crop_start + self.crop_size)]
            dist_masks = dist_masks[:, crop_start:(crop_start + self.crop_size), crop_start:(crop_start + self.crop_size)]
            
            ddists = self._get_ddists(dists)
            dangles = self._get_dangles(angles)
            
            yield seqs, evos, dangles, ddists, dist_masks
    
    @staticmethod
    def _get_ddists(dists):
        bins = np.linspace(2, 22, 63)
        ddists = torch.tensor(np.searchsorted(bins, dists.numpy()))

        return ddists.float()
    
    @staticmethod
    def _get_dangles(angles):
        bins = np.linspace(-np.pi, np.pi, 36)
        dangles = torch.tensor(np.searchsorted(bins, angles.numpy()))
        dangles = dangles[:, 0, :] * dangles[:, 1, :]

        return dangles
            
            
if __name__ == '__main__':
    loader = Part1Dataloader('val', 16, 256)
    for seq, evo, dangles, dist, dist_mask in tqdm(loader):
        print(seq.shape)
