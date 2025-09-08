from pathlib import Path

import numpy as np
from skimage.io import imsave

from network.loss import Loss
from utils.base_utils import color_map_backward
from utils.draw_utils import concat_images_list
from skimage.metrics import structural_similarity
import trimesh


def compute_psnr(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

def process_key_img(data, h, w):
    img = color_map_backward(data.detach().cpu().numpy())
    img = img.reshape([h, w, -1])
    if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
    return img

def get_key_images(data_pr, keys, h, w):
    results=[]
    for k in keys:
        if k in data_pr: results.append(process_key_img(data_pr[k], h, w))
    return results

def draw_materials(data_pr, h, w):
    keys=['diffuse_albedo', 'diffuse_light', 'diffuse_color', 'albedo',
          'specular_albedo', 'specular_light', 'specular_color', 'specular_direct_light',
          'metallic', 'roughness', 'occ_prob', 'indirect_light']
    results = get_key_images(data_pr, keys, h, w)
    results = [concat_images_list(*results[0:4]),concat_images_list(*results[3:7]),concat_images_list(*results[7:])]
    return results

class ShapeRenderMetrics(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgb_gt = color_map_backward(data_pr['gt_rgb'].detach().cpu().numpy()) # h,w,3
        imgs = [rgb_gt]

        # compute psnr
        rgb_pr = color_map_backward(data_pr['ray_rgb'].detach().cpu().numpy())  # h,w,3
        psnr = compute_psnr(rgb_gt, rgb_pr)
        ssim = structural_similarity(rgb_gt, rgb_pr, win_size=11, channel_axis=2, data_range=255)
        outputs={'psnr': np.asarray([psnr]),'ssim': np.asarray([ssim])}
        imgs.append(rgb_pr)

        # normal
        h, w, _ = rgb_pr.shape
        normal = color_map_backward(data_pr['normal_vis'].detach().cpu().numpy())  # h,w,3
        imgs.append(normal.reshape([h,w,3]))

        if 'human_light' in data_pr:
            imgs.append(process_key_img(data_pr['human_light'], h, w))

        if 'radiance' in data_pr:
            imgs.append(color_map_backward(data_pr['radiance'].detach().cpu().numpy())) # h,w,3
        elif 'acc' in data_pr:
            imgs.append(process_key_img(data_pr['acc'], h, w))

        imgs = [concat_images_list(*imgs)]

        imgs += draw_materials(data_pr, h, w)

        # output image
        data_index=kwargs['data_index']
        model_name=kwargs['model_name']
        output_path = Path(f'data/train_vis/{model_name}')
        output_path.mkdir(exist_ok=True, parents=True)
        imsave(f'{str(output_path)}/{step}-index-{data_index}.jpg', concat_images_list(*imgs, vert=True))
        
        if 'vertices' in data_pr:
            mesh = trimesh.Trimesh(data_pr['vertices'], data_pr['triangles'])
            mesh.export(f'{str(output_path)}/{step}-mesh.ply')
        return outputs

class MaterialRenderMetrics(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        data_pr['error'] = (data_pr['rgb_gt'] - data_pr['rgb_pr']).abs()
        rgb_gt = color_map_backward(data_pr['rgb_gt'].detach().cpu().numpy()) # h,w,3
        rgb_pr = color_map_backward(data_pr['rgb_pr'].detach().cpu().numpy())  # h,w,3
        imgs = [rgb_gt, rgb_pr]
        if 'rgb_pr_nis' in data_pr:
            data_pr['error_nis'] = (data_pr['rgb_gt'] - data_pr['rgb_pr_nis']).abs()
            rgb_pr_nis = color_map_backward(data_pr['rgb_pr_nis'].detach().cpu().numpy())  # h,w,3
            imgs_nis = [rgb_gt, rgb_pr_nis]
            
        # compute psnr
        psnr = compute_psnr(rgb_gt, rgb_pr)
        ssim = structural_similarity(rgb_gt, rgb_pr, win_size=11, channel_axis=2, data_range=255)
        
        outputs={'psnr': np.asarray([psnr]),'ssim': np.asarray([ssim])}
        if 'rgb_pr_nis' in data_pr:
            psnr_nis = compute_psnr(rgb_gt, rgb_pr_nis)
            ssim_nis = structural_similarity(rgb_gt, rgb_pr_nis, win_size=11, channel_axis=2, data_range=255)
            outputs['psnr_nis'] = np.asarray([psnr_nis])
            outputs['ssim_nis'] = np.asarray([ssim_nis])
            print(f'\npsnr: {psnr}, psnr_nis: {psnr_nis}')

        additional_keys = ['albedo', 'metallic', 'roughness', 'normal', 'diffuse_color', 'specular_color', 'error', 'diffuse_color', 'occ_trace', 'indirect_light']
        for k in additional_keys:
            img = color_map_backward(data_pr[k].detach().cpu().numpy())
            if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
            imgs.append(img)
        output_imgs = [concat_images_list(*imgs[:6]),concat_images_list(*imgs[6:])]
        
        # output image
        data_index=kwargs['data_index']
        model_name=kwargs['model_name']
        output_path = Path(f'data/train_vis/{model_name}')
        output_path.mkdir(exist_ok=True, parents=True)
        imsave(f'{str(output_path)}/{step}-index-{data_index}.jpg', concat_images_list(*output_imgs, vert=True))
        
        if 'rgb_pr_nis' in data_pr:
            for k in additional_keys:
                img = color_map_backward(data_pr[k+'_nis'].detach().cpu().numpy())
                if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
                imgs_nis.append(img)
            output_imgs_nis = [concat_images_list(*imgs_nis[:6]),concat_images_list(*imgs_nis[6:])]
            imsave(f'{str(output_path)}/{step}-index-{data_index}-nis.jpg', concat_images_list(*output_imgs_nis, vert=True))

        if 'envmap' in data_pr:
            envmap = color_map_backward(data_pr['envmap'].detach().cpu().numpy())
            imsave(f'{str(output_path)}/{step}-envmap.jpg', envmap)
        return outputs

name2metrics={
    'shape_render': ShapeRenderMetrics,
    'mat_render': MaterialRenderMetrics,
}

def psnr(results):
    return np.mean(results['psnr'])

def psnr_nis(results):
    return np.mean(results['psnr_nis'])


name2key_metrics={
    'psnr': psnr,
    'psnr_nis': psnr_nis,
}
