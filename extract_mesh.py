import argparse
from pathlib import Path
import sys

import torch
import trimesh
from network.invRenderer import name2renderer
from utils.base_utils import load_cfg
from utils.network_utils import extract_geometry
from omegaconf import OmegaConf

def main():
    cfg = load_cfg(flags.cfg)
    cfg = OmegaConf.create(cfg)
    unknown_cfg = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, unknown_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if flags.extra_name is not None:
        cfg['name'] = '_'.join([cfg['name'], flags.extra_name])

    if flags.model == 'best':
        ckpt = torch.load(f'data/model/{cfg["name"]}/model_best.pth')
    else:
        ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    step = ckpt['step']
    kwargs = ckpt['kwargs']
    
    cfg.update(kwargs)
    network = name2renderer[cfg['network']](cfg, training=False)
    network.load_ckpt(ckpt)
    network.eval().cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f'successfully load {cfg["name"]} step {step}!')

    bbox_min = -torch.ones(3)
    bbox_max = torch.ones(3)
    ratio = cfg['blend_ratio']
    print(f'Blend ratio: {ratio}')
    with torch.no_grad():
        vertices, triangles = extract_geometry(bbox_min, bbox_max, flags.resolution, 0, lambda x: network.sdf_network.sdf(x, torch.full(x.shape[:-1] + (1, ), ratio)))

    # output geometry
    mesh = trimesh.Trimesh(vertices, triangles)
    output_dir = Path('data/meshes')
    output_dir.mkdir(exist_ok=True)
    mesh.export(str(output_dir/f'{cfg["name"]}-{step}.ply'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/shape/syn/lego.yaml')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--model', type=str, choices=['latest', 'best'], default='latest')
    parser.add_argument('-e', '--extra_name', type=str, default=None)
    flags, unknown = parser.parse_known_args()
    main()