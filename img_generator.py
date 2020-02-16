#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/14
"""
import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import utils
from dlib_alignment import face_recover, dlib_detect_face
from models.SRGAN_model import SRGANModel
from my_utils.project_utils import mkdir_if_not_exist, traverse_dir_files
from root_dir import DATA_DIR


class ImgGenerator(object):
    """
    超分辨率生成图像
    """

    def __init__(self):
        # self.pretrain_model_G = os.path.join(DATA_DIR, 'models', '90000_G.pth')
        self.pretrain_model_G = os.path.join(DATA_DIR, 'models', '200000_G.pth')

        self.transform = self.get_transforms()
        self.sr_model = self.init_model()

    def sr_forward(self, img, padding=0.5, moving=0.1):
        """
        超分辨率，图像
        """
        img_aligned, M = dlib_detect_face(img, padding=padding, image_size=(128, 128), moving=moving)
        input_img = torch.unsqueeze(self.transform(Image.fromarray(img_aligned)), 0)
        self.sr_model.var_L = input_img.to(self.sr_model.device)
        self.sr_model.test()
        output_img = self.sr_model.fake_H.squeeze(0).cpu().numpy()
        output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        rec_img = face_recover(output_img, M * 4, img)

        return rec_img

    def init_model(self):
        """
        初始化模型
        """
        sr_model = SRGANModel(self.get_FaceSR_opt(), is_train=False)
        sr_model.load()
        print('[Info] device: {}'.format(sr_model.device))

        return sr_model

    def get_transforms(self):
        """
        正则化
        """
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])
        return transform

    def get_FaceSR_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu_ids', type=str, default="0")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr_G', type=float, default=1e-4)
        parser.add_argument('--weight_decay_G', type=float, default=0)
        parser.add_argument('--beta1_G', type=float, default=0.9)
        parser.add_argument('--beta2_G', type=float, default=0.99)
        parser.add_argument('--lr_D', type=float, default=1e-4)
        parser.add_argument('--weight_decay_D', type=float, default=0)
        parser.add_argument('--beta1_D', type=float, default=0.9)
        parser.add_argument('--beta2_D', type=float, default=0.99)
        parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
        parser.add_argument('--niter', type=int, default=100000)
        parser.add_argument('--warmup_iter', type=int, default=-1)
        parser.add_argument('--lr_steps', type=list, default=[50000])
        parser.add_argument('--lr_gamma', type=float, default=0.5)
        parser.add_argument('--pixel_criterion', type=str, default='l1')
        parser.add_argument('--pixel_weight', type=float, default=1e-2)
        parser.add_argument('--feature_criterion', type=str, default='l1')
        parser.add_argument('--feature_weight', type=float, default=1)
        parser.add_argument('--gan_type', type=str, default='ragan')
        parser.add_argument('--gan_weight', type=float, default=5e-3)
        parser.add_argument('--D_update_ratio', type=int, default=1)
        parser.add_argument('--D_init_iters', type=int, default=0)

        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--val_freq', type=int, default=1000)
        parser.add_argument('--save_freq', type=int, default=10000)
        parser.add_argument('--crop_size', type=float, default=0.85)
        parser.add_argument('--lr_size', type=int, default=128)
        parser.add_argument('--hr_size', type=int, default=512)

        # network G
        parser.add_argument('--which_model_G', type=str, default='RRDBNet')
        parser.add_argument('--G_in_nc', type=int, default=3)
        parser.add_argument('--out_nc', type=int, default=3)
        parser.add_argument('--G_nf', type=int, default=64)
        parser.add_argument('--nb', type=int, default=16)

        # network D
        parser.add_argument('--which_model_D', type=str, default='discriminator_vgg_128')
        parser.add_argument('--D_in_nc', type=int, default=3)
        parser.add_argument('--D_nf', type=int, default=64)

        # data dir
        parser.add_argument('--pretrain_model_G', type=str, default='90000_G.pth')
        parser.add_argument('--pretrain_model_D', type=str, default=None)

        args = parser.parse_args()

        args.pretrain_model_G = self.pretrain_model_G

        return args


def generate_img(img_path, out_path):
    print('[Info] 图像路径: {}'.format(img_path))
    # 图像
    img = utils.read_cv2_img(img_path)
    img_name = img_path.split('/')[-1].split('.')[0]

    # 输出文件夹
    outs_dir = os.path.join(DATA_DIR, 'outs')
    mkdir_if_not_exist(outs_dir)

    ig = ImgGenerator()

    rec_img = ig.sr_forward(img=img)  # 超分辨率图像

    # 输出图像
    # out_path = os.path.join(outs_dir, '{}.output.2.jpg'.format(img_name))
    utils.save_image(rec_img, out_path)
    # print('[Info] 输出路径: {}'.format(out_path))


def generate_img_dir(img_dir, out_dir):
    """
    处理视频文件夹，输出至文件夹
    """
    # img_dir = os.path.join(DATA_DIR, 'imgs')
    # out_dir = os.path.join(DATA_DIR, 'outs')
    mkdir_if_not_exist(out_dir)
    paths_list, names_list = traverse_dir_files(img_dir)

    ig = ImgGenerator()

    count = 0
    for path, name in zip(paths_list, names_list):
        print('[Info] 图像路径: {}'.format(path))
        # img_path = os.path.join(DATA_DIR, 'imgs', 'input_4.jpg')
        sr_name = name.split('.')[0] + '.jpg'
        out_path = os.path.join(out_dir, sr_name)

        try:
            img = utils.read_cv2_img(path)
            rec_img = ig.sr_forward(img=img)  # 超分辨率图像
            utils.save_image(rec_img, out_path)
        except Exception as e:
            print('[Info] 异常: {}'.format(e))
            print('[Info] 图像: {}'.format(path))

        count += 1
        if count % 100 == 0:
            print('[Info] count {}'.format(count))

    print('[Info] 视频文件处理完成!')
    print('[Info] 输出文件夹: {}'.format(out_dir))


def generate_img_dir_test():
    # img_dir = os.path.join(DATA_DIR, 'imgs')
    # out_dir = os.path.join(DATA_DIR, 'outs')
    img_dir = os.path.join('/mydata/workspace/datasets/s2a4zhengsheng/trainA')
    out_dir = os.path.join('/mydata/workspace/datasets/s2a4zhengsheng/trainA_sr')
    generate_img_dir(img_dir, out_dir)


def main():
    generate_img_dir_test()


if __name__ == '__main__':
    main()
