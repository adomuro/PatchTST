import math
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as opt
import pytorch_lightning as pl

from .loss import MaskedMSELoss, MaskedL1Loss, MaskedHuberLoss, IQRLoss
from utils.stats import estimate_noise


# Define the Transformer model for denoising pre-training
class TransformerDenoiser(nn.Module):
    def __init__(self, n_dim, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerDenoiser, self).__init__()
        self.n_dim = n_dim
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers
        )
        # Add a fully connected layer for reconstruction
        self.reconstruction_layer = nn.Linear(d_model, n_dim)

    def forward(self, src, tgt):
        # src: input sequence (corrupted time series data)
        # tgt: target sequence (original non-noisy time series data)
        output = self.transformer(src, tgt)
        # Pass the decoder output through the reconstruction layer
        output = self.reconstruction_layer(output) 
        return output
    

def create_noise_mask(size, p):
    mask = torch.rand(size) < p
    return mask.float()

# Function to add noise to the input data
def add_noise_to_input(input_data, noise_probability):
    noise_mask = create_noise_mask(input_data.size(), noise_probability)
    noisy_input = input_data.clone()
    noisy_input[noise_mask] = 0.0
    return noisy_input

class LitImputer(pl.LightningModule):
    def __init__(self,
                 c_in:int,
                 d_model=128,
                 nhead=16,
                 dim_feedforward=128,
                 eye=0,
                 dropout:float=0.,
                 num_layers=3,
                 lr=0.001,
                 learned_pos=False,
                 norm='batch',
                 attention='full',
                 seq_len=None,
                 keep_ratio=0.,
                 random_ratio=1.,
                 token_ratio=0.,
                 uniform_bound=2.,
                 train_unit='standard',
                 train_loss='mae',
                 **kwargs
                 ):
        """Instanciate a Lit TPT imputer module

        Args:
            c_in (int, optional): number of input dimensions. Defaults to 1.
            d_model (int, optional): Encoder latent dimension. Defaults to 128.
            nhead (int, optional): Number of heads. Defaults to 8.
            dim_feedforward (int, optional): number of feedforward units in the encoder.
                Defaults to 256.
            dropout (float, optional): Encoder dropout. Defaults to 0.1.
            num_layers (int, optional): Number of encoder layer(s). Defaults to 3.
            lr (float, optional): AdamW earning rate. Defaults to 0.001.
        """
        super().__init__()
        self.save_hyperparameters()
        self.c_in = c_in
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.lr = lr
        self.norm = norm
        # self.zero_ratio =  zero_ratio
        self.keep_ratio = keep_ratio
        self.random_ratio = random_ratio
        self.uniform_bound = uniform_bound
        self.token_ratio = token_ratio
        self.train_unit = train_unit
        self.imputing_losses = []
        self.bias_denoising_losses = []
        self.training_losses = []

        assert train_unit in ['standard', 'noise', 'flux', 'star']

        self.ie = nn.Linear(c_in, d_model)
        self.pe = PosEmbedding(d_model, learned=learned_pos)
        self.ea = EyeAttention(eye)
        if attention == 'linear':
            self.encoder = Linformer(
                dim=d_model,
                seq_len=seq_len,
                depth=num_layers,
                heads=nhead,
                k=32,
                one_kv_head=True,
                share_kv=True
            )
        else:
            encoder_layer = TransformerEncoderLayer(d_model,
                                                    nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=0.1,
                                                    batch_first=True,
                                                    norm=norm, seq_len=seq_len,
                                                    attention=attention
                                                    )
            self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.recons_head = nn.Linear(d_model, c_in)
        self.msk_token_emb = nn.Parameter(torch.randn(1, 1, d_model))

        if train_loss == 'mse':
            self.criterion = MaskedMSELoss()  # masked or not masked
        elif train_loss == 'mae':
            self.criterion = MaskedL1Loss()
        elif train_loss == 'huber':
            self.criterion = MaskedHuberLoss()
        else:
            raise NotImplementedError
        self.mae_loss = MaskedL1Loss()
        self.mse_loss = MaskedMSELoss()
        self.iqr_loss = IQRLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitImputer")
        parser.add_argument("--c_in", type=int)
        parser.add_argument("--d_model", type=int)
        parser.add_argument("--nhead", type=int)
        parser.add_argument("--dim_feedforward", type=int)
        parser.add_argument("--eye", type=int)
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--num_layers", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--learned_pos", action='store_true')
        parser.add_argument("--norm", type=str)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--seq_len", type=int)
        parser.add_argument("--keep_ratio", type=float)
        parser.add_argument("--random_ratio", type=float)
        parser.add_argument("--token_ratio", type=float)
        parser.add_argument("--uniform_bound", type=float)
        parser.add_argument("--train_unit", type=str,
                            choices=['standard', 'noise', 'star'])
        parser.add_argument("--train_loss", type=str,
                            choices=['mae', 'mse', 'huber'])
        return parent_parser

    def configure_optimizers(self):
        optimiser = opt.Adam(self.parameters(), lr=self.lr)
        return optimiser

    def apply_mask(self, x, mask):
        if mask is None:
            out = x
            out[torch.isnan(out)] = 0.
            return out, torch.zeros_like(x)

        r = torch.rand_like(x)
        keep_mask = (~mask | (r <= self.keep_ratio)).to(x.dtype)
        random_mask = (mask & (self.keep_ratio < r)
                       & (r <= self.keep_ratio+self.random_ratio)).to(x.dtype)
        token_mask = (mask & ((1-self.token_ratio) < r)).to(x.dtype)
        xm, xM = -self.uniform_bound, self.uniform_bound
        out = x * keep_mask + (torch.rand_like(x)*(xM-xm)+xm) * random_mask
        out[torch.isnan(out)] = 0.
        return out, token_mask

    def forward(self, x, mask=None):
        out, token_mask = self.apply_mask(x, mask)
        out = self.ie(out)
        if self.token_ratio:
            out = self.msk_token_emb * token_mask + (1-token_mask) * out

        out = out + self.pe(out)

        attention_mask = self.ea(x)
        out = self.encoder(out, mask=attention_mask)
        out = self.recons_head(out)
        return out

    def get_attention_maps(self, x, mask=None):
        out, token_mask = self.apply_mask(x, mask)
        out = self.ie(out)
        if self.token_ratio:
            out = self.msk_token_emb * token_mask + (1-token_mask) * out

        out = out + self.pe(out)

        attention_mask = self.ea(x)
        out = self.encoder.get_attention_maps(out, mask=attention_mask)
        return out

    def training_step(self, batch, batch_index):

        x, y, m, info = batch
        pred = self.forward(x, m)

        if self.train_unit == 'standard':
            loss = self.criterion(pred, y, m)
        elif self.train_unit == 'noise':
            noise = estimate_noise(y)
            loss = self.criterion(pred/noise, y/noise, m)
        elif self.train_unit == 'star':
            y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
            pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
            y_d = detrend(y_o, pred_o)
            loss = self.criterion(y_d, torch.ones_like(y_d), m)
        if torch.isnan(loss):
            print('Pred has nans?', torch.isnan(pred).sum().item())
            print('Y has nans?', torch.isnan(
                y).sum().item(), f' shape({y.shape})')
            print('M has fully masked items?',
                  ((m.int()-1).sum((1, 2)) == 0).sum().item())
            print('mu has nans?', torch.isnan(info['mu']).sum().item())
            raise ValueError('Nan Loss found during training')
        
        self.training_losses.append(loss.item())
        return {'loss': loss}

    def on_train_epoch_end(self):
        # Compute average training loss
        training_loss = torch.tensor(self.training_losses).mean()

        # Log or use the average training loss as needed
        #print("Training Loss:", training_loss)
        self.log('train_loss', training_loss)

    def validation_step(self, batch, batch_index, dataloader_idx=None):
        variable_noise = 0.5
        x, y, m, info = batch
        pred = self.forward(x, m)

        noise = estimate_noise(y)
        variable = (noise <= variable_noise).squeeze()
        n_variables = variable.sum()
        pred_noise = pred / noise
        y_noise = y / noise

        # star normalised unit space
        y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
        pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
        y_d = detrend(y_o, pred_o)

        out = dict()
        if dataloader_idx is None or dataloader_idx == 0:  # Imputing
            # Imputation
            rmse = torch.sqrt(self.mse_loss(pred, y, m))
            rmse_noise = torch.sqrt(self.mse_loss(pred_noise, y_noise, m))
            rmse_star = torch.sqrt(self.mse_loss(torch.ones_like(y_d), y_d, m))
            mae = self.mae_loss(pred, y, m)
            mae_noise = self.mae_loss(pred_noise, y_noise, m)
            mae_star = self.mae_loss(torch.ones_like(y_d), y_d, m)

            self.imputing_losses.append({
            'rmse': rmse.item(),
            'rmse_noise': rmse_noise.item(),
            'rmse_star': rmse_star.item(),
            'mae': mae.item(),
            'mae_noise': mae_noise.item(),
            'mae_star': mae_star.item()
        })
            
            out.update({'val_mrmse': rmse, 'val_mmae': mae,
                        'val_mrmse_noise': rmse_noise, 'val_mmae_noise': mae_noise,
                        'val_mrmse_star': rmse_star, 'val_mmae_star': mae_star
                        })

        if dataloader_idx is None or dataloader_idx == 1:
            # Bias
            rmse = torch.sqrt(self.mse_loss(pred, y))
            rmse_noise = torch.sqrt(self.mse_loss(pred_noise, y_noise))
            rmse_star = torch.sqrt(self.mse_loss(torch.ones_like(y_d), y_d))
            mae = self.mae_loss(pred, y)
            mae_noise = self.mae_loss(pred_noise, y_noise)
            mae_star = self.mae_loss(torch.ones_like(y_d), y_d)

            out.update({'val_rmse': rmse, 'val_mae': mae,
                        'val_rmse_noise': rmse_noise, 'val_mae_noise': mae_noise,
                        'val_rmse_star': rmse_star, 'val_mae_star': mae_star
                        })

            # Denoising
            iqr = self.iqr_loss(pred, y)
            iqr_variable = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable = self.iqr_loss((pred-y)[variable])
            iqr_noise = self.iqr_loss(pred_noise, y_noise)
            iqr_variable_noise = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable_noise = self.iqr_loss(
                    (pred_noise-y_noise)[variable])
            iqr_star = self.iqr_loss(y_d)
            iqr_variable_star = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable_star = self.iqr_loss(y_d[variable])
            

            self.bias_denoising_losses.append({
            'rmse': rmse.item(),
            'rmse_noise': rmse_noise.item(),
            'rmse_star': rmse_star.item(),
            'mae': mae.item(),
            'mae_noise': mae_noise.item(),
            'mae_star': mae_star.item(),
            'iqr': iqr.item(),
            'iqr_variable': iqr_variable.item(),
            'iqr_noise': iqr_noise.item(),
            'iqr_variable_noise': iqr_variable_noise.item(),
            'iqr_star': iqr_star.item(),
            'iqr_variable_star': iqr_variable_star.item()
            })

            out.update({'val_IQR': iqr, 'val_IQR_var': iqr_variable,
                        'val_IQR_noise': iqr_noise, 'val_IQR_var_noise': iqr_variable_noise,
                        'val_IQR_star': iqr_star, 'val_IQR_var_star': iqr_variable_star,
                        })
        
        # Accumulate the loss
        #loss = out['val_mrmse']  # Choose the appropriate loss here
        #self.validation_losses.append(loss.item())
    
        #self.validation_step_outputs.append(self.out)
        
        return out


    def on_validation_epoch_end(self):
        # Compute the average loss from the accumulated losses
        #avg_loss = torch.mean(torch.tensor(self.validation_losses))
        #self.log('val_loss', avg_loss, prog_bar=True)

        imputing_losses = self.imputing_losses
        bias_denoising_losses = self.bias_denoising_losses

        # Compute average losses for each phase
        imputing_avg_losses = {
            'rmse': torch.tensor([l['rmse'] for l in imputing_losses]).mean(),
            'rmse_noise': torch.tensor([l['rmse_noise'] for l in imputing_losses]).mean(),
            'rmse_star': torch.tensor([l['rmse_star'] for l in imputing_losses]).mean(),
            'mae': torch.tensor([l['mae'] for l in imputing_losses]).mean(),
            'mae_noise': torch.tensor([l['mae_noise'] for l in imputing_losses]).mean(),
            'mae_star': torch.tensor([l['mae_star'] for l in imputing_losses]).mean()
        }

        bias_denoising_avg_losses = {
            'rmse': torch.tensor([l['rmse'] for l in bias_denoising_losses]).mean(),
            'rmse_noise': torch.tensor([l['rmse_noise'] for l in bias_denoising_losses]).mean(),
            'rmse_star': torch.tensor([l['rmse_star'] for l in bias_denoising_losses]).mean(),
            'mae': torch.tensor([l['mae'] for l in bias_denoising_losses]).mean(),
            'mae_noise': torch.tensor([l['mae_noise'] for l in bias_denoising_losses]).mean(),
            'mae_star': torch.tensor([l['mae_star'] for l in bias_denoising_losses]).mean(),
            'iqr': torch.tensor([l['iqr'] for l in bias_denoising_losses]).mean(),
            'iqr_variable': torch.tensor([l['iqr_variable'] for l in bias_denoising_losses]).mean(),
            'iqr_noise': torch.tensor([l['iqr_noise'] for l in bias_denoising_losses]).mean(),
            'iqr_variable_noise': torch.tensor([l['iqr_variable_noise'] for l in bias_denoising_losses]).mean(),
            'iqr_star': torch.tensor([l['iqr_star'] for l in bias_denoising_losses]).mean(),
            'iqr_variable_star': torch.tensor([l['iqr_variable_star'] for l in bias_denoising_losses]).mean()
        }
        # Log or use the average losses as needed
        #print("Imputing Avg Losses:", imputing_avg_losses)
        #print("Bias/Denoising Avg Losses:", bias_denoising_avg_losses)
        self.log('val_loss_imputation', imputing_avg_losses['rmse'], prog_bar=True)
        self.log('val_IQRLoss_denoising', imputing_avg_losses['rmse'], prog_bar=True)


    def test_step(self, batch, batch_index, dataloader_idx=None):
        d_out = self.validation_step(batch, batch_index, dataloader_idx)
        return {k.replace('val', 'test'): v for k, v in d_out.items()}

    def on_test_epoch_end(self, outputs):
        return self.on_validation_epoch_end()


class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """

    def __init__(self, dim, dropout, E_proj, F_proj, causal_mask, full_attention=False):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask
        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        # Instead of classic masking, we have to do this, because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:, :, None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:, :, None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1, 2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(K.device)
                K = torch.matmul(K, self.E)
            else:
                K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q / \
            torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(Q.device)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, float('-inf'))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1, 2)
            if self.is_proj_tensor:
                self.F = self.F.to(V.device)
                V = torch.matmul(V, self.F)
            else:
                V = self.F(V)
            V = V.transpose(1, 2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor

    # import torch
# from linformer import LinformerSelfAttention


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x,
                src_mask=None, src_key_padding_mask=None,
                context=None, **kwargs):
        if src_mask is not None:
            warnings.warn('SRC MASK not used in Linformer')
        if src_key_padding_mask is not None:
            warnings.warn('src_key_padding_mask not used in Linformer')
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        def proj_seq_len(args): return torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (
            self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        def merge_key_values(t): return t.reshape(
            b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route={}, layer_dropout=0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values(
        )), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: (
                {key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args},
                                  {**g_args, **new_g_args})
    return routed_args


class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, reversible=False, dropout=0.):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LinformerSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head,
                                          one_kv_head=one_kv_head, share_kv=share_kv, dropout=dropout)
            ff = FeedForward(dim, dropout=dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x,
                mask=None
                ):
        return self.net(x)

    import torch.nn.functional as F


def eye_large(n, width=1, dtype=torch.float32, device=None):
    out = np.eye(n)
    for k in range(1, width):
        out += np.eye(n, k=k)
        out += np.eye(n, k=-k)
    return torch.tensor(out, dtype=dtype, device=device)


class EyeAttention(nn.Module):
    def __init__(self, width=1, max_len=500):
        super(EyeAttention, self).__init__()
        # "don't attend now"
        if width == 1:
            mask = torch.eye(max_len, dtype=bool)
        elif width == 0:
            mask = None
        else:
            mask = eye_large(max_len, width=width, dtype=bool)
        self.register_buffer('mask', mask)

    def forward(self, x):
        if self.mask is None:
            return None
        return self.mask[:x.shape[1], :x.shape[1]]


class PosEmbedding(nn.Module):
    def __init__(self, d_model, learned=False, max_len=5000, dtype=torch.float32):
        super(PosEmbedding, self).__init__()
        if learned:
            self.pe = LearnedPosEmbedding(
                d_model, max_len=max_len, dtype=dtype)
        else:
            self.pe = FixedPosEmbedding(d_model, max_len=max_len, dtype=dtype)

    def forward(self, x):
        return self.pe(x)


class FixedPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000, dtype=torch.float32):
        super(FixedPosEmbedding, self).__init__()
        # Compute the positional embeddings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        pe.requires_grad = False

        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=dtype)
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LearnedPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000, dtype=torch.float32):
        super(LearnedPosEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model, dtype=dtype))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required). (B, L, D)
        Shape:
            output: tensor of shape (B, L, D)
        """
        return self.pe[:, :x.size(1), :]


class BatchNorm(nn.BatchNorm1d):
    """Overrides nn.BatchNorm1d to define shape structure identical
    to LayerNorm, i.e. (N, L, C) and not (N, C, L)"""

    def forward(self, input):
        return super().forward(input.transpose(1, 2)).transpose(1, 2)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Overrides nn.TransformerEncoderLayer class with
    - BatchNorm option as suggested by Zerveas et al https://arxiv.org/abs/2010.02803
    - PrboSparse attention from Zhou et al 
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 norm='layer', attention='full', seq_len=None,
                 device=None, dtype=None) -> None:
        # this combination of shapes hasn't been dealt with yet
        assert batch_first or norm == 'layer'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,
                                                      layer_norm_eps, batch_first, norm_first, device, dtype)

        if attention == 'full':
            pass
        else:
            raise NotImplementedError
        if norm == 'layer':
            pass
        elif norm == 'batch':
            self.norm1 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            raise NotImplementedError


class TransformerEncoder(nn.TransformerEncoder):
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, x, x, attn_mask=mask)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


def inverse_standardise_batch(x, mu, sigma):
    return x * sigma + mu


def detrend(x, trend):
    return x / trend