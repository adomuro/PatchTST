
import torch
from torch import nn
import pandas as pd

from .core import Callback

# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride, indicator):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride
        self.indicator = indicator

    def before_forward(self): self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        
        #self.xb = x # change!!!!!!!!!!
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        
        ### CHANGE ####
        if self.indicator == 0:
            self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]           

        if self.indicator == 1:
            ind =  torch.zeros(xb_patch.shape[0], xb_patch.shape[1], xb_patch.shape[2], 1)
            xb_patch = torch.cat((xb_patch, ind), dim=3)
            # learner get the transformed input
            self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len+1]           

class PatchNoiseCB(Callback):

    def __init__(self, patch_len, stride, indicator):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride
        self.indicator = indicator

    def before_forward(self): self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        mu = 0
        std = 1
        xb_Noise = addGaussianNoise(xb_patch, mu, std)
        
        if self.indicator == 0:
            self.learner.xb = xb_Noise               # learner.xb: noisy 4D tensor [bs x num_patch x n_vars x patch_len]   
            self.learner.yb = xb_patch               # learner.yb: non-noisy 4d tensor
        
        if self.indicator == 1:
        
            ind =  torch.zeros(xb_patch.shape[0], xb_patch.shape[1], xb_patch.shape[2], 1)
            xb_Noise = torch.cat((xb_Noise, ind), dim=3)
            xb_patch = torch.cat((xb_patch, ind), dim=3)
            # learner get the transformed input
            self.learner.xb = xb_Noise               # learner.xb: noisy 4D tensor with Indicator Variable [bs x num_patch x n_vars x patch_len]   
            self.learner.yb = xb_patch               # learner.yb: non-noisy 4d tensor with Indicator Variable

class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio, mask_type, indicator,
                        mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type 
        self.indicator = indicator  

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss        
        device = self.learner.device       
 
    def before_forward(self): self.patch_masking()
        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        if self.mask_type == 'random_masking':
            xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio, self.indicator)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        if self.mask_type == 'patches':
            xb_mask, _, self.mask = masking_patches(xb_patch, self.mask_ratio)
        if self.mask_type == 'features':
            xb_mask, _, self.mask = masking_features(xb_patch, self.mask_ratio)
        if self.mask_type == 'same_mask':
            xb_mask, _, self.mask, _ = masking_same(xb_patch, self.mask_ratio)

        
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor   
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
 
    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """

        #### TYPE OF MASK NEED DIFFERENT LOSSES???? #####

        if self.indicator == 0:
            
            # Calculate the average of each masked patch
            mask = self.mask.unsqueeze(-1)
            patch_averages = (target * mask).sum(dim=-1) / target.shape[3]

            # Calculate loss
            loss = (preds - target) ** 2
            #loss = loss.mean(dim=-2)
            loss = loss.mean(dim=-1) # RANDOM MASKING
            loss = loss + patch_averages
            #loss = (loss * self.mask[0][0][0]).sum() / self.mask.sum()
            loss = (loss * self.mask).sum() / self.mask.sum()

            
        
        if self.indicator == 1:

            # Calculate the average of each masked patch
            mask = self.mask.unsqueeze(-1)
            patch_averages = (target * mask).sum(dim=-1) / target.shape[3]

            # Calculate loss
            indicator =  torch.zeros(target.shape[0], target.shape[1], target.shape[2], 1)
            target_2 = torch.cat((target, indicator), dim=3)

            loss = (preds - target_2) ** 2
            #loss = loss.mean(dim=-2)
            loss = loss.mean(dim=-1) # RANDOM MASKING
            loss = loss + patch_averages
            #loss = (loss * self.mask[0][0][0]).sum() / self.mask.sum()
            loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                              # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)             # xb: [bs x num_patch x n_vars x patch_len]
    
    return xb, num_patch

def addGaussianNoise(x, mu, std):
    noise = torch.randn(x.size()) * std + mu
    x_noisy = x + noise
    return x_noisy

class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x

def random_masking(xb, mask_ratio, indicator):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio)) # number of patches that will be kept in the output tensor x_kept as an int

    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove, tensor that contains the indices of the sorted noise tensor in ascending order along the second dimension
       
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars], tensor containing the indices that will restore the original order of ids_shuffle
    
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]    # ids_keep: [bs x len_keep x nvars], the first len_keep are selected from ids_shuffle
    
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len], the corresponding patches from x are gathered

    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len], a tensor created with zeros to represent the patches that will be removed, it has a size of the mask ratio

    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len], x_kept and x_removed are concatenated along the secnond dimension to form a new tensor
    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len], tensor x_ is gathered back in its original order specified by ids_restore to obtain the output tensor x_masked
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars], creates a tensor initialized to all ones, which will be used to represent areas to keep
    mask[:, :len_keep, :] = 0 # the first len_keep elements in the second dimension of the tensor mask to 0, this corresponds to the first subset of patches that were kept earlier in the code

    # unshuffle to get the binary mask, tensor with original order where 1s are masked elements
    mask = torch.gather(mask, dim=1, index=ids_restore)   # [bs x num_patch x nvars], rearrange the tensor mask using the indices in ids_restore. This unshuffles the tensor to match the original order of patches in the input. The resulting tensor has same dims as mask but with elements rearranged based on the ids of ids_restore
    
    if indicator == 0:
        return x_masked, x_kept, mask, ids_restore
    if indicator == 1: 
        mask_2 = mask.unsqueeze(-1)
        x_indicator = torch.cat((x_masked, mask_2), dim=3)

        return x_indicator, x_kept, mask, ids_restore

def masking_patches(xb, mask_ratio):
    # Last `mask_ratio´ of patches are masked 
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio)) # number of patches that will be kept in the output tensor x_kept as an int  
    # keep the first subsets
    x_kept = x[:, :len_keep, :]    # x_kept: [bs x len_keep x nvars], the first len_keep are selected from x

    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)     # x_removed: [bs x (L-len_keep) x nvars x patch_len], a tensor created with zeros to represent the patches that will be removed, it has a size of the mask ratio

    x_masked = torch.cat([x_kept, x_removed], dim=1)       

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars], creates a tensor initialized to all ones, which will be used to represent areas to keep
    mask[:, :len_keep, :] = 0 # the first len_keep elements in the second dimension of the tensor mask to 0, this corresponds to the first subset of patches that were kept earlier in the code
    return x_masked, x_kept, mask


def masking_features(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = round(nvars * (1 - mask_ratio)) # number of features that will be kept in the output tensor x_kept as an int
    # keep the first subset
    x_kept = x[:, :, :len_keep]    # ids_keep: [bs x len_keep x nvars], the first len_keep are selected from ids_shuffle
    # removed x
    x_removed = torch.zeros(bs, L, nvars-len_keep, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len], a tensor created with zeros to represent the patches that will be removed, it has a size of the mask ratio
    x_masked = torch.cat([x_kept, x_removed], dim=2)                                          # x_: [bs x L x nvars x patch_len], x_kept and x_removed are concatenated along the secnond dimension to form a new tensor
    
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars], creates a tensor initialized to all ones, which will be used to represent areas to keep
    mask[:, :, :len_keep] = 0 # the first len_keep elements in the second dimension of the tensor mask to 0, this corresponds to the first subset of patches that were kept earlier in the code

    return x_masked, x_kept, mask

def masking_same(xb, mask_ratio):
     # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(nvars * (1 - mask_ratio)) # number of patches that will be kept in the output tensor x_kept as an int

    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise[0][0]) 

    ids_restore = torch.argsort(ids_shuffle)  

    # keep the first subset
    ids_keep = ids_shuffle[:len_keep, ] 
    
    x_kept = torch.index_select(x, dim=2, index=ids_keep)     
    # removed x
    x_removed = torch.zeros(bs, L, nvars-len_keep, D, device=xb.device)   
    
    x_ = torch.cat([x_kept, x_removed], dim=2)
    # combine the kept part and the removed one
    x_masked = torch.index_select(x_, dim=2, index=ids_restore)
    
    x_aux = torch.zeros(bs, L, nvars, device=xb.device) 
    mask_kept = torch.index_select(x_aux, dim=2, index=ids_keep)    
    # removed x
    mask_removed = torch.ones(bs, L, nvars-len_keep, device=xb.device)               
    mask_x_ = torch.cat([mask_kept, mask_removed], dim=2)                                       
    # combine the kept part and the removed one
    mask =  torch.index_select(mask_x_, dim=2, index=ids_restore) 

    return x_masked, x_kept, mask, ids_restore

def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2,20,4,5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
    breakpoint()


