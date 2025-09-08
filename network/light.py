import numpy as np
import torch
import imageio
from . import renderutils as ru
from .light_utils import *


class EnvLight(torch.nn.Module):

    def __init__(self, path=None, device=None, scale=1.0, min_res=16, start_res=16, max_res=512, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable
        self.start_res = start_res

        # init an empty cubemap
        self.base = torch.nn.Parameter(
            torch.full((6, self.max_res, self.max_res, 3), np.log(0.5), dtype=torch.float32, device=self.device),
            # torch.rand((6, self.max_res, self.max_res, 3),dtype=torch.float32, device=self.device).log(),
            requires_grad=self.trainable,
        )
        
        # try to load from file
        if path is not None:
            self.load(path)
        
        # self.build_mips()
        self.level = max(0, int(np.log2(self.max_res / self.start_res))+0.5)
    
    def upsample(self):
        if self.level > 0:
            # import pdb;pdb.set_trace()
            self.level = max(self.level-1, 0)

    def load(self, path):
        # load latlong env map from file
        image = imageio.imread(path)
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255

        image = torch.from_numpy(image).to(self.device) * self.scale
        cubemap = latlong_to_cubemap(image, [self.max_res, self.max_res], self.device)

        self.base.data = cubemap


    def build_mips(self, cutoff=0.99):
        
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def build_mips_direct(self, cutoff=0.99):
        
        self.base_mip = [self.base]
        while self.base_mip[-1].shape[1] > self.min_res:
            self.base_mip += [cubemap_mip.apply(self.base_mip[-1])]

    def get_mip(self, roughness):
        # map roughness to mip_level (float):
        # roughness: 0 --> self.min_roughness --> self.max_roughness --> 1
        # mip_level: 0 --> 0                  --> M - 2              --> M - 1
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
        

    def get_mip_direct(self, roughness):
        beta = 0.9
        tan_theta_m = roughness * np.sqrt(beta / (1-beta))  # roughness is already squared
        tan_theta_h = 2 * tan_theta_m / (1 - tan_theta_m**2)
        theta_h = torch.atan(tan_theta_h)
        level = (theta_h/np.arctan(1/(self.max_res/2))).log().clamp(0, len(self.base_mip) - 1)
        # level = (tan_theta_h * 0.5 * self.max_res * 2).log().clamp(0, len(self.base_mip) - 1)
        level = torch.full_like(level, 2.0)
        return level        

    

    def __call__(self, l, roughness=None):
        # l: [..., 3], normalized direction pointing from shading position to light
        # roughness: [..., 1]

        prefix = l.shape[:-1]
        if len(prefix) != 3: # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if roughness is None:
            # diffuse light
            light = dr.texture(self.diffuse[None, ...], l, filter_mode='linear', boundary_mode='cube')
        else:
            # specular light
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )

        light = light.view(*prefix, -1)
        
        return torch.exp(light)
    
    
    def direct_light(self, l, roughness=None):
        # roughness = None
        # l: [..., 3], normalized direction pointing from shading position to light

        prefix = l.shape[:-1]
        if len(prefix) != 3: # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        # if roughness is None:
        #     light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        # else:
        #     # # # specular light
        #     # miplevel = self.get_mip_direct(roughness)
        #     # # light = dr.texture(
        #     # #     self.base_mip[0][None, ...], 
        #     # #     l,
        #     # #     mip=list(m[None, ...] for m in self.base_mip[1:]), 
        #     # #     mip_level_bias=miplevel[..., 0], 
        #     # #     filter_mode='linear-mipmap-linear', 
        #     # #     boundary_mode='cube'
        #     # # )
        #     # self.level = 0
        #     light = dr.texture(
        #         self.base[None, ...], 
        #         l,
        #         mip=list(m[None, ...] for m in self.base_mip[1:]), 
        #         mip_level_bias=torch.zeros_like(roughness[..., 0]), 
        #         filter_mode='linear-mipmap-linear', 
        #         boundary_mode='cube'
        #     )
        #     # import pdb;pdb.set_trace()
            
        light = light.view(*prefix, -1)
        
        return torch.exp(light)