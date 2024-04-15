import torch
import torch.nn as nn

from .spiral import SpiralBlock, SpiralResBlock
from .spiral import MeshSamplingCSR as MeshSampling

from .common import InstanceNorm

class RegisteredMeshEncoder(nn.Module):
    def __init__(self, 
        dim_I,                                  # dim of Indentity code
        dim_E,                                  # dim of Expression code
        spiral_indices,                         # spiral indices
        downsample_matrices,                    # downsample matrix list
        channel_in      = 3,                    # input channel (3 for xyz)
        mid_channels    = [16, 32, 64, 64, 64], # channels for each block
        feature_channel = 128,                  # dim of feature
        shared_backbone = False                 # whether the backbone is shared for Identity and Expression
        ):
        super().__init__()

        self.shared_backbone = shared_backbone

        channels = [channel_in] + mid_channels

        norm_layer = InstanceNorm
        activation = nn.ELU

        fc_in_channel = downsample_matrices[1].shape[0]*channels[5]

        def make_backbone(fc_out_channel):
            bone = nn.Sequential(
                # (B, NV, C)
                SpiralBlock(channels[0], channels[1], spiral_indices[0], norm_layer, activation),
                MeshSampling(downsample_matrices[0]),
                # (B, NV/4, C1)

                SpiralBlock(channels[1], channels[2], spiral_indices[1], norm_layer, activation),
                MeshSampling(downsample_matrices[1]),
                # (B, NV/16, C2)

                SpiralBlock(   channels[2], channels[3], spiral_indices[2], norm_layer, activation),
                SpiralResBlock(channels[3], channels[4], spiral_indices[2], norm_layer, activation),
                SpiralResBlock(channels[4], channels[5], spiral_indices[2], norm_layer, activation),
                # (B, NV/16, C5)

                nn.Flatten(1, -1),
                # (B, NV/16*C5)
                nn.Linear(fc_in_channel, fc_out_channel)
                # (B, C_out)
            )
            return bone
        
        if self.shared_backbone:
            self.backbone = make_backbone(feature_channel)
        else:
            self.backbone_ind = make_backbone(feature_channel)
            self.backbone_exp = make_backbone(feature_channel)
        
        self.head_ind = nn.Sequential(
            activation(),
            nn.Linear(feature_channel, dim_I)
        )
        self.head_exp = nn.Sequential(
            activation(),
            nn.Linear(feature_channel, dim_E)
        )
    
    def forward(self, mesh_v):
        if self.shared_backbone:
            feat_ind = feat_exp = self.backbone(mesh_v)
        else:
            feat_ind = self.backbone_ind(mesh_v)
            feat_exp = self.backbone_exp(mesh_v)

        vec_ind = self.head_ind(feat_ind)
        vec_exp = self.head_exp(feat_exp)

        return vec_ind, vec_exp
