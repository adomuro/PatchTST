
import torch
from torch import nn
import pandas as pd

from .core import Callback

# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]           

class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
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
    
        ##### RANDOM MASKING 
        #xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        ##### MASKING THE LAST VALUES OF THE PATCH
        xb_mask, _, self.mask = masking_patch_forecast(xb_patch, self.mask_ratio)
        
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor   
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
 
    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-2)
        # loss = loss.mean(dim=-1) # RANDOM MASKING
        loss = (loss * self.mask[0][0][0]).sum() / self.mask.sum()
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


class MaskCB(Callback):
    def __init__(self, mask_ratio, mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        #self.patch_len = patch_len
        #self.stride = stride
        self.mask_ratio = mask_ratio    

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss        
        device = self.learner.device       
 
    def before_forward(self): self.masking()
        
    def masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
    
        ##### RANDOM MASKING 
        #xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        ##### MASKING THE LAST VALUES OF THE PATCH

        #xb = self.xb.transpose(1, 2)
        self.learner.yb = self.xb
        xb_mask, _, self.mask = masking_forecast(self.xb, self.mask_ratio)
        
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.mask = self.mask    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor  
 
    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask[:,:, 0]).sum() / self.mask.sum()
        return loss

def random_masking(xb, mask_ratio):
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

    return x_masked, x_kept, mask, ids_restore

def masking_patch_forecast(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(round(D * (1 - mask_ratio))) # number of patches that will be kept in the output tensor x_kept as an int

    # keep the first subset
    x_kept = x[:, :, :, :len_keep]    # ids_keep: [bs x len_keep x nvars], the first len_keep are selected from ids_shuffle
    #x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len], the corresponding patches from x are gathered

    # removed x
    x_removed = torch.zeros(bs, L, nvars, D-len_keep, device=xb.device)                      # x_removed: [bs x (L-len_keep) x nvars x patch_len], a tensor created with zeros to represent the patches that will be removed, it has a size of the mask ratio
    x_masked = torch.cat([x_kept, x_removed], dim=3)                                         # x_: [bs x L x nvars x patch_len], x_kept and x_removed are concatenated along the second dimension to form a new tensor
    
    # combine the kept part and the removed one
    #x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len], tensor x_ is gathered back in its original order specified by ids_restore to obtain the output tensor x_masked

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars, D], device=x.device)  # mask: [bs x num_patch x nvars], creates a tensor initialized to all ones, which will be used to represent areas to keep
    mask[:, :, :, :len_keep] = 0 # the first len_keep elements in the second dimension of the tensor mask to 0, this corresponds to the first subset of patches that were kept earlier in the code

    return x_masked, x_kept, mask #, ids_restore

def masking_forecast(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, D, nvars = xb.shape
    x = xb.clone()
    
    len_keep = int(round(D * (1 - mask_ratio))) # number of patches that will be kept in the output tensor x_kept as an int
    # keep the first subset
    x_kept = x[:, :len_keep, :]    # ids_keep: [bs x len_keep x nvars], the first len_keep are selected from ids_shuffle
    #x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len], the corresponding patches from x are gathered

    # removed x
    x_removed = torch.zeros(bs, D-len_keep, nvars, device=xb.device)                      # x_removed: [bs x (L-len_keep) x nvars x patch_len], a tensor created with zeros to represent the patches that will be removed, it has a size of the mask ratio
    x_masked = torch.cat([x_kept, x_removed], dim=1)                                         # x_: [bs x L x nvars x patch_len], x_kept and x_removed are concatenated along the second dimension to form a new tensor


    # combine the kept part and the removed one
    #x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len], tensor x_ is gathered back in its original order specified by ids_restore to obtain the output tensor x_masked

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, D, nvars], device=x.device)  # mask: [bs x num_patch x nvars], creates a tensor initialized to all ones, which will be used to represent areas to keep
    mask[:, :len_keep, :] = 0 # the first len_keep elements in the second dimension of the tensor mask to 0, this corresponds to the first subset of patches that were kept earlier in the code

    return x_masked, x_kept, mask #, ids_restore


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


