
import torch
import torch.nn as nn

class NaiveVAEs(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mu_logvar):
        mu, lv = torch.tensor_split(mu_logvar, 2, dim=-1) # mean, log var

        if self.training:
            vec = self.draw_from_gaussian(mu, lv)
        else:
            vec = mu
        return vec, (mu, lv)
    
    @staticmethod
    def draw_from_gaussian(mu, lv):
        return  mu + torch.exp(lv/2)*torch.randn_like(mu)
        