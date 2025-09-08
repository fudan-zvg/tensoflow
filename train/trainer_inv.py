import os
import random
from pathlib import Path
from shutil import copyfile
import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.name2dataset import name2dataset
from dataset.database import parse_database_name
from network.loss import name2loss
from network.invRenderer import name2renderer
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import dummy_collate_fn

import matplotlib.pyplot as plt
import cv2
from utils.base_utils import Timing, color_map_backward
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from eval_geo import normalize_numpy

class TrainerInv:
    default_cfg={
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_xyz_init" : 1e-2, 
        "lr_net_init" : 1e-3,
        "lr_env_init" : 1e-2,
        "lr_decay_target_ratio" : 5e-2,
        "lr_decay_iters" : -1,
        "total_step": 200000,
        "train_log_step": 20,
        "val_interval": 10000,
        "test": False,
        "test_interval": 10000,
        "save_interval": 500,
        "worker_num": 8,
        'random_seed': 6033,
        'recording': [ './', './train', './network', './dataset'],
        'isMaterial': False,

        # network setting
        'device' : 'cuda',
        # 'gridSize' : [128, 128, 128]
        'N_voxel_init': 2097152, # 128**3
        'N_voxel_final': 64000000, # 400**3
        'aabb' : [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        'step_ratio' : 0.5,
        'alphaMask_thres' : 0.0001,
        'marched_weights_thres' : 0.0001,
        'sdf_n_comp' : 16,
        'app_n_comp' : 36,
        'sdf_dim' : 128,
        'app_dim' : 128,
        'upsample_list' : None,
        'update_AlphaMask_lst' : None,
        'hessian_update_list': None,
        'sparse_update_list': None,
        'sample_level_step': None,
        'has_radiance_field': False,
        'radiance_field_step': 0,
        'scratch': True,
        'ckpt_path': None,
    }
    def _init_dataset(self):
        self.train_set=name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True, self.cfg['dataset_dir'])
        self.train_set=DataLoader(self.train_set,1,True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
        print(f'train set len {len(self.train_set)}')
        self.val_set_list, self.val_set_names = [], []
        for val_set_cfg in self.cfg['val_set_list']:
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            val_set = name2dataset[val_type](val_cfg, False, self.cfg['dataset_dir'])
            val_set = DataLoader(val_set,1,False,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
            self.val_set_list.append(val_set)
            self.val_set_names.append(name)
            print(f'{name} val set len {len(val_set)}')

    def _init_test_dataset(self):
        self.test_database = parse_database_name(self.cfg['database_name'], self.cfg['dataset_dir'], isTest=True, isWhiteBG=self.cfg['isBGWhite'])
        self.test_ids = self.test_database.get_img_ids()
        self.test_dataloader = DataLoader(self.test_ids, 1, False, num_workers=self.cfg['worker_num'])
        print(f'Test set len {len(self.test_ids)}')
    
    def _init_network(self):
        if self.cfg['optimizer_type']=='adam':
            self.optimizer_type = Adam
        elif self.cfg['optimizer_type']=='sgd':
            self.optimizer_type = SGD
        else:
            raise NotImplementedError

        best_para,start_step=0,0
        if (os.path.exists(self.pth_fn) and not self.cfg['scratch']) or self.cfg['ckpt_path'] is not None:
            if self.cfg['ckpt_path'] is not None:
                checkpoint=torch.load(self.cfg['ckpt_path'], weights_only=False)
            else:
                checkpoint=torch.load(self.pth_fn, weights_only=False)
            checkpoint['network_state_dict'] = {k:v for k, v in checkpoint['network_state_dict'].items() if 'flow' not in k}
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            if 'kwargs' in checkpoint:
                kwargs = checkpoint['kwargs']
                self.cfg.update(kwargs)
            self.N_voxel_list = checkpoint['N_voxel_list']
            self.network = name2renderer[self.cfg['network']](self.cfg).cuda()
            self.network.load_ckpt(checkpoint)
            grad_vars = self.network.get_train_opt_params(self.cfg['lr_xyz_init'], self.cfg['lr_net_init'], self.cfg['lr_env_init'])
            self.optimizer = self.optimizer_type(grad_vars, betas=(0.9, 0.99))
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.cur_lr_xyz, self.cur_lr_net = checkpoint['lr_xyz'], checkpoint['lr_net']
            self.lr_factor, self.pre_lr_factor = checkpoint['lr_factor'], checkpoint['pre_lr_factor']
            print(f'==> resuming from step {start_step} best para {best_para}')
        else:
            self.N_voxel_list = (np.round(np.exp(
                np.linspace(np.log(self.cfg['N_voxel_init']), np.log(self.cfg['N_voxel_final']), len(self.cfg['upsample_list']) + 1 if self.cfg['upsample_list'] is not None else 1))).astype(np.int32)).tolist()
            n_voxels = self.N_voxel_list.pop(0)
            self.cfg['gridSize'] = self.N_to_reso(n_voxels, self.cfg['aabb'])
            self.network = name2renderer[self.cfg['network']](self.cfg).cuda()
            grad_vars = self.network.get_train_opt_params(self.cfg['lr_xyz_init'], self.cfg['lr_net_init'], self.cfg['lr_env_init'])
            self.optimizer = self.optimizer_type(grad_vars, betas=(0.9, 0.99))
            self.cur_lr_xyz, self.cur_lr_net = self.cfg['lr_xyz_init'], self.cfg['lr_net_init']
            self.lr_factor = self.pre_lr_factor = 1.0
            self.file_backup()

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        self.val_evaluator=ValidationEvaluator(self.cfg)
        print(f"Model results will be saved to data/model/{self.model_name}")
        print(f"NVS results will be saved to data/nvs/{self.model_name}")
        print(f"Visualization results will be saved to data/train_vis/{self.model_name}-val")

        if self.cfg['lr_decay_iters'] < 0:
            self.cfg['lr_decay_iters'] = self.cfg['total_step']
        
        if not self.cfg['isMaterial']:
            print(f"lr_decay_target_ratio : {self.cfg['lr_decay_target_ratio']}, lr_decay_iters : {self.cfg['lr_decay_iters']}, \
                upsample list : {self.cfg['upsample_list']}, occ_loss_step : {self.cfg['occ_loss_step']}, radiance_field_step: {self.cfg['radiance_field_step']}")
        return best_para, start_step

    def __init__(self,cfg, config_path=None):
        self.config_path = config_path
        self.cfg={**self.default_cfg,**cfg}
        self.cfg['hessian_update_list'] = self.cfg['hessian_update_list'] if self.cfg['hessian_update_list'] is not None else self.cfg['upsample_list']
        self.cfg['sparse_update_list'] = self.cfg['sparse_update_list'] if self.cfg['sparse_update_list'] is not None else self.cfg['upsample_list']
        torch.manual_seed(self.cfg['random_seed'])
        np.random.seed(self.cfg['random_seed'])
        random.seed(self.cfg['random_seed'])
        self.model_name=cfg['name']
        self.model_dir=os.path.join('data/model', cfg['name'])
        if not os.path.exists(self.model_dir): Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

    def run(self):
        # torch.autograd.set_detect_anomaly(True)
        self._init_dataset()
        if self.cfg['test']:
            self._init_test_dataset()
        self._init_logger()
        best_para, start_step = self._init_network()

        train_iter=iter(self.train_set)

        pbar=tqdm(total=self.cfg['total_step'],bar_format='{r_bar}')
        pbar.update(start_step)
        for step in range(start_step,self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step']=step

            self.network.train()
            self.optimizer.zero_grad()
            self.network.zero_grad()

            log_info={}
            outputs=self.network(train_data)
            for loss in self.val_losses:
                loss_results = loss(outputs,train_data,step)
                for k,v in loss_results.items():
                    log_info[k]=v

            loss=0
            for k,v in log_info.items():
                if k.startswith('loss'):
                    assert not torch.any(torch.isnan(v)), f"{k} is nan!!!!"
                    loss=loss+torch.mean(v)

            # assert not torch.any(torch.isnan(loss)), "loss is nan!!!!"
            # with torch.autograd.detect_anomaly():
            loss.backward()
            self.optimizer.step()

            if ((step+1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info,step+1,'train')

            if (step+1)%self.cfg['val_interval']==0 or (step+1)==self.cfg['total_step'] or step == start_step:
                torch.cuda.empty_cache()
                if not self.cfg['isMaterial']:            
                    self.network.color_network.envlight.build_mips()
                val_results={}
                val_para = 0
                for vi, val_set in enumerate(self.val_set_list):
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        val_results[f'{self.val_set_names[vi]}-{k}'] = v
                    # always use the final val set to select model!
                    val_para = val_para_cur

                if val_para>best_para:
                    print(f'Step {step+1} -- New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para=val_para
                    self._save_model(step+1,best_para,self.best_pth_fn)
                self._log_data(val_results,step+1,'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if self.cfg['test'] and ((step+1)%self.cfg['test_interval']==0 or (step+1)==self.cfg['total_step']):
                self.run_test(step)
                
                
            if (step+1)%self.cfg['save_interval']==0:
                save_fn = None
                self._save_model(step+1,best_para,save_fn=save_fn)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_factor
            
            self.cur_lr_xyz *= self.lr_factor
            self.cur_lr_net *= self.lr_factor   
            self.update_learning_rate(step)
            if not self.cfg['isMaterial']:            
                pbar.set_postfix(
                    loss=float(loss.detach().cpu().numpy()), \
                    eik=float(torch.mean(log_info['loss_eikonal'])), \
                    inv_s=float(self.network.deviation_network.variance), \
                    # spa=float(torch.mean(log_info['loss_sparse'])), \
                    # tv=float(torch.mean(log_info['loss_tv_sdf'])), \
                    # rad=float(torch.mean(log_info['loss_radiance']) if 'loss_radiance' in log_info else 0. ), \
                    # gau=float(torch.mean(log_info['loss_gaussian']) if 'loss_gaussian' in log_info else 0. ), \
                    # hes=float(torch.mean(log_info['loss_hessian']) if 'loss_hessian' in log_info else 0. ), \
                    # mask=float(torch.mean(log_info['loss_mask']) if 'loss_mask' in log_info else 0. ), \
                    col=float(torch.mean(log_info['loss_rgb'])), \
                    num=outputs['sample_num'], \
                    psnr=float(torch.mean(outputs['psnr'])), \
                    # lr_x=self.cur_lr_xyz, lr_n=self.cur_lr_net,
                    )

                pbar.update(1)

                if self.network.occ_grid is None and self.cfg['update_AlphaMask_lst'] is not None and step in self.cfg['update_AlphaMask_lst']:
                    new_aabb = self.network.updateAlphaMask()
                    # if step == self.cfg['update_AlphaMask_lst'][0]:
                    #     self.network.sdf_network.shrink(new_aabb)

                    # if step == self.cfg['update_AlphaMask_lst'][1]:
                        # self.network.filtering_train_rays()
                        # self.network._shuffle_train_batch()
                if self.cfg['sample_level_step'] is not None and step > self.cfg['sample_level_step']:
                    self.network.sample_level = True

                if self.cfg['upsample_list'] is not None and step in self.cfg['upsample_list']:
                    # upsamp_gridSize = 2 * self.network.gridSize
                    n_voxels = self.N_voxel_list.pop(0)
                    upsamp_gridSize = self.N_to_reso(n_voxels, self.cfg['aabb'])
                    self.network.upsample_sdf_grid(upsamp_gridSize)
                    
                    grad_vars = self.network.get_train_opt_params(self.cfg['lr_xyz_init'], self.cfg['lr_net_init'], self.cfg['lr_env_init'])
                    self.optimizer = self.optimizer_type(grad_vars, betas=(0.9, 0.99))
                    self.cur_lr_xyz, self.cur_lr_net = self.cfg['lr_xyz_init'] * 0.5, self.cfg['lr_net_init']
            else:
                pbar.set_postfix(
                    loss=float(loss.detach().cpu().numpy()), \
                    col=float(torch.mean(log_info['loss_rgb'])), \
                    mat_reg=float(torch.mean(log_info['loss_mat_reg'])), \
                    light_reg=float(torch.mean(log_info['loss_diffuse_light'])), \
                    nis=float(torch.mean(log_info['loss_nis'])), \
                    var=float(torch.mean(log_info['variance'])), \
                    psnr=float(torch.mean(outputs['psnr'])), \
                    lr_net=self.cur_lr_net)
                pbar.update(1)    
            
            del loss, log_info
        pbar.close()

    def run_test(self, step=0):
        num = len(self.test_ids)
        img_avg_psnr = 0
        img_avg_ssim = 0
        normal_avg_mae = 0
        for _, ind in tqdm(enumerate(self.test_ids)):
            pose = self.test_database.get_pose(ind)
            K = self.test_database.get_K(ind)
            gt_imgs = self.test_database.get_image(ind)
            h, w = gt_imgs.shape[:2]
            with torch.no_grad():
                outputs = self.network.nvs(pose, K, h, w)
            for k in outputs:
                if k != 'normal':
                    outputs[k] = color_map_backward(outputs[k])
            cur_psnr = psnr(gt_imgs, outputs['color'])
            cur_ssim = ssim(gt_imgs, outputs['color'], win_size=11, channel_axis=2, data_range=255)  
            img_avg_psnr += cur_psnr
            img_avg_ssim += cur_ssim
            
            gt_normals = self.test_database.get_normal(ind)
            gt_normals = normalize_numpy(gt_normals)
            cur_mae = np.mean(np.arccos(np.clip(np.sum(gt_normals * outputs['normal'], axis=-1), -1, 1)) * 180 / np.pi)
            normal_avg_mae += cur_mae
            
        img_avg_psnr /= num
        img_avg_ssim /= num
        normal_avg_mae /= num

        log_results = {'PSNR': img_avg_psnr, 'SSIM': img_avg_ssim, 'Normal_MAE': normal_avg_mae}
        self.logger.log(log_results, 'test', step, verbose=True)  
    
    def update_learning_rate(self, step):
        progress = step / self.cfg['lr_decay_iters']
        cur_lr_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - self.cfg['lr_decay_target_ratio']) + self.cfg['lr_decay_target_ratio']
        self.lr_factor = cur_lr_factor / self.pre_lr_factor
        self.pre_lr_factor = cur_lr_factor

    def checkAlphaMask(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

    def N_to_reso(self, n_voxels, bbox):
        bbox = torch.tensor(bbox)
        xyz_min, xyz_max = bbox
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)   # total volumes / number
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = self.pth_fn if save_fn is None else save_fn
        state_dict = {
            'step':step,
            'best_para':best_para,
            'lr_factor': self.lr_factor,
            'pre_lr_factor': self.pre_lr_factor,
            'lr_xyz': self.cur_lr_xyz, 
            'lr_net': self.cur_lr_net,
            'optimizer_state_dict': self.optimizer.state_dict(),       
            'N_voxel_list': self.N_voxel_list,
        } 
        state_dict.update(self.network.ckpt_to_save())
        torch.save(state_dict,save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)

    def file_backup(self):
        dir_lis = self.cfg['recording']
        os.makedirs(os.path.join(self.model_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.model_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.config_path, os.path.join(self.model_dir, 'recording', 'config.yaml'))


    def draw_level_grid(self, sdf_fun, save_dir, iter_step='gt', resolution=512):
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        Z = torch.ones(X.shape) * 0.4

        pos = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        level = sdf_fun(pos.to('cuda')).reshape(resolution, resolution).cpu().detach().numpy()

        X, Y = X.numpy(), Y.numpy()
        plt.figure(figsize=(8, 8), dpi=200)
        cs = plt.contour(X, Y, level, 20, alpha=.75)
        plt.clabel(cs, inline=True, fontsize=8)
        plt.savefig(os.path.join(save_dir, 'test_level_{}.png'.format(iter_step)))
        cv2.imwrite(os.path.join(save_dir, 'test_sdf_{}.exr'.format(iter_step)), level[..., None])