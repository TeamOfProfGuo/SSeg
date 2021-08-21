
from torch import nn
from addict import Dict

from model.net import Encoder, Centerpiece, Decoder

class Base_Net(nn.Module):

    def __init__(self, n_classes, config={}):
        super().__init__()

        self.config = Dict(config)
        
        self.encoder = Encoder(
            encoder=config.general.encoder,
            encoder_args=config.encoder_args
        )

        self.cp = Centerpiece(
            cp=config.general.cp, 
            feats=config.general.feats,
            cp_args=config.cp_args if config.general.cp != 'none' else {}
        )

        self.decoder = Decoder(
            n_classes=n_classes,
            feats=config.general.feats,
            aux=config.decoder_args.aux,
            final_aux=config.decoder_args.final_aux,
            final_fuse=config.decoder_args.final_fuse,
            lf_args=config.decoder_args.lf_args,
            final_args=config.decoder_args.final_args
        )

    def forward(self, l, d):
        feats = self.encoder(l, d)    
        feats = self.cp(feats)
        out_feats = self.decoder(feats)   
        return tuple(out_feats)

def get_basenet(dataset='nyud', config={}):
    from .datasets import datasets
    model = Base_Net(datasets[dataset.lower()].NUM_CLASS, config=config)
    return model