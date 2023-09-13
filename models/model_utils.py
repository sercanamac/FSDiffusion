from termcolor import colored
import torch

from models.networks.vqvae_networks.network import VQVAE
from models.networks.pvqvae_networks.auto_encoder import PVQVAE

def load_vqvae(vq_conf, vq_ckpt, opt=None):
    assert type(vq_ckpt) == str
    vq_ckpt = "vqvae_epoch-best.pth"
    # init vqvae for decoding shapes
    mparam = vq_conf.model.params
    n_embed = mparam.n_embed
    embed_dim = mparam.embed_dim
    ddconfig = mparam.ddconfig

    n_down = len(ddconfig.ch_mult) - 1

    vqvae = VQVAE(ddconfig, n_embed, embed_dim)
    
    map_fn = lambda storage, loc: storage
    state_dict = torch.load(vq_ckpt, map_location=map_fn)
    if 'vqvae' in state_dict:
        vqvae.load_state_dict(state_dict['vqvae'])
    else:
        vqvae.load_state_dict(state_dict)

    print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))
    vqvae.requires_grad = False

    vqvae.to(opt.device)
    vqvae.eval()
    return vqvae

def load_pvqvae(vq_conf, vq_ckpt, opt=None):
    assert type(vq_ckpt) == str
    vq_ckpt = "/home/fslsegment/sercan/FSDiffusion/logs_home/2023-09-10T02-50-01-pvqvae-snet-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_epoch-best.pth"
    # init vqvae for decoding shapes
    mparam = vq_conf.model.params
    n_embed = mparam.n_embed
    embed_dim = mparam.embed_dim
    ddconfig = mparam.ddconfig
    
    n_down = len(ddconfig.ch_mult) - 1

    vqvae = PVQVAE(ddconfig, n_embed, embed_dim)
    
    map_fn = lambda storage, loc: storage
    state_dict = torch.load(vq_ckpt, map_location=map_fn)
    if 'vqvae' in state_dict:
        vqvae.load_state_dict(state_dict['vqvae'])
    else:
        vqvae.load_state_dict(state_dict)

    print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))
    vqvae.requires_grad = False

    vqvae.to(opt.device)
    vqvae.eval()
    return vqvae