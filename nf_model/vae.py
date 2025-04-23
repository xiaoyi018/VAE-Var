import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from networks.LGUnet_all import LGUnet_all_1
from networks_old.transformer import LGUnet_all
import torch.utils.checkpoint as cp
import yaml

class VAE(nn.Module):
    def __init__(self, latent_channel_list, **params):
        super(VAE, self).__init__()
        
        latent_channel_list_db = [elem*2 for elem in latent_channel_list]
        # self.param_encoder = copy.copy(params)
        # self.param_encoder["outchans_list"] = latent_channel_list_db
        # self.param_decoder = copy.copy(params)
        # self.param_decoder["inchans_list"] = latent_channel_list
        # self.param_decoder["outchans_list"] = copy.copy(params["inchans_list"])

        with open("nf_model/parameters1.yaml", 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

        self.param_encoder = cfg_params["encoder"]
        self.param_decoder = cfg_params["decoder"]
        
        print(self.param_encoder)
        print(self.param_decoder)
        
        self.enc = LGUnet_all_1(**self.param_encoder)
        self.dec = LGUnet_all_1(**self.param_decoder)
        
    def encoder(self, x):
        encoded = self.enc(x)
        # encoded = cp.checkpoint(self.enc, x)

        return encoded.chunk(2, dim = 1)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def decoder(self, z):
        # return cp.checkpoint(self.dec, z)
        return self.dec(z)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

class VAE_lr(nn.Module):
    def __init__(self, param_path, lora_rank=0):
        super(VAE_lr, self).__init__()
    
        with open("nf_model/%s.yaml"%param_path, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

        self.param_encoder = cfg_params["encoder"]
        self.param_decoder = cfg_params["decoder"]

        self.param_encoder["rank"] = lora_rank
        self.param_decoder["rank"] = lora_rank
        
        print(self.param_encoder)
        print(self.param_decoder)
        
        self.enc = LGUnet_all(**self.param_encoder)
        self.dec = LGUnet_all(**self.param_decoder)
        
    def encoder(self, x):
        encoded = self.enc(x)
        # encoded = cp.checkpoint(self.enc, x)

        return encoded.chunk(2, dim = 1)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        # return cp.checkpoint(self.dec, z)
        return self.dec(z)
   
    def decoder_hr(self, z):
        # return cp.checkpoint(self.dec, z)
        x = self.dec(z)
        return F.interpolate(x, (721, 1440))

    def finetune(self,):
        for name, param in self.named_parameters():
            if name.split(".")[-2] in ["kA", "kB", "qA", "qB", "vA", "vB"]:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def loss_function(recon_x, x, mu, log_var, sigma):
    MSE = torch.sum((recon_x - x)**2 ) / (2 * sigma**2) #* (32 / imsize)**2
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD, MSE*2 * sigma**2, KLD
