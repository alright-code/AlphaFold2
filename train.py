import torch
import torch.nn.functional as F
import torch.optim as optim

from dataloader import Part1Dataloader
from model import Part1Model


# Model Params
NUM_BLOCKS = 48
C_M = 384
H_M = 2
W_M = 256
A_M = 32
HEADS_M = 8
C_Z = 128
H_Z = 256
W_Z = 256
A_Z = 128
HEADS_Z = 4
C_S = 384

# Data Params
BATCH_SIZE = 16
CROP_SIZE = 256

# Train Params
EPOCHS = 10
CHECKPOINT_FREQUENCY = 2
LR = 1e-3


def main():
    model = Part1Model(num_blocks=NUM_BLOCKS,
                       c_m=C_M,
                       h_m=H_M,
                       w_m=W_M,
                       a_m=A_M,
                       heads_m=HEADS_M,
                       c_z=C_Z,
                       h_z=H_Z,
                       w_z=W_Z,
                       a_z=A_Z,
                       heads_z=HEADS_Z,
                       c_s=C_S)
    model.cuda()
    
    train_loader = Part1Dataloader(mode='train', batch_size=BATCH_SIZE, crop_size=CROP_SIZE)
    val_loader = Part1Dataloader(mode='val', batch_size=BATCH_SIZE, crop_size=CROP_SIZE)
    
    optimizer = optim.Adam(model.parameters(), LR)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for seq, evo, angle, dist, mask in train_loader:
            seq = seq.cuda()
            evo = evo.cuda()
            dist = dist.cuda()
            angle = angle.cuda()
            mask = mask.cuda()
            
            dist_pred, angle_pred = model(seq, evo)
            
            dist_loss = F.cross_entropy(dist_pred, dist, reduction='none').mul_(mask).sum().div_(mask.sum())
            angle_loss = F.cross_entropy(angle_pred, angle, reduction='none').mul_(mask).sum().div_(mask.sum())
            loss = dist_loss + angle_loss
            train_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inpt, dist, angle, mask in val_loader:
                inpt = inpt.cuda()
                dist = dist.cuda()
                angle = angle.cuda()
                mask = mask.cuda()
                
                dist_pred, angle_pred = model(inpt)
            
                dist_loss = F.cross_entropy(dist_pred, dist, reduction='none').mul_(mask).sum().div_(mask.sum())
                angle_loss = F.cross_entropy(angle_pred, angle, reduction='none').mul_(mask).sum().div_(mask.sum())
                loss = dist_loss + angle_loss
                val_loss += loss
                
        train_loss = train_loss.item() / len(train_loader)
        val_loss = val_loss.item() / len(val_loader)
        
        print(f'Epoch {epoch} Train Loss: {train_loss:.2f}')
        print(f'Epoch {epoch} Validation Loss: {val_loss:.2f}')
        
        if epoch % CHECKPOINT_FREQUENCY == 0:
            torch.save(model.state_dict(), f'{epoch}-{val_loss:.2f}')


if __name__ == '__main__':
    main()
