#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/16
"""

import os
import sys
import shutil
import random

p = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))

if p not in sys.path:
    sys.path.append(p)

from my_utils.project_utils import traverse_dir_files, mkdir_if_not_exist


def process_s2azhengsheng():
    """
    处理s2azhengsheng数据集
    """
    trainA_dir = "/mydata/workspace/datasets/s2a4zhengsheng/trainA_sr"
    trainB_dir = "/mydata/workspace/datasets/s2a4zhengsheng/trainB"
    testA_dir = "/mydata/workspace/datasets/s2a4zhengsheng/testA_sr"
    testB_dir = "/mydata/workspace/datasets/s2a4zhengsheng/testB"

    trainA_out_dir = "/mydata/workspace/datasets/s2a4zhengsheng_sr/trainA"
    trainB_out_dir = "/mydata/workspace/datasets/s2a4zhengsheng_sr/trainB"
    testA_out_dir = "/mydata/workspace/datasets/s2a4zhengsheng_sr/testA"
    testB_out_dir = "/mydata/workspace/datasets/s2a4zhengsheng_sr/testB"

    process_dir(trainA_dir, trainA_out_dir, 70000, "p")
    process_dir(trainB_dir, trainB_out_dir, 70000, "c")
    process_dir(testA_dir, testA_out_dir, 4500, "p")
    process_dir(testB_dir, testB_out_dir, 4500, "c")

    print('[Info] 全部处理完成!')


def process_dir(src_dir, dst_dir, num, type):
    """
    处理文件夹
    :param src_dir: 源文件夹
    :param dst_dir: 目标文件夹
    :param num: 数量
    :return: 复制
    """
    print('[Info] 源文件夹: {}'.format(src_dir))
    print('[Info] 目标文件夹: {}'.format(dst_dir))
    mkdir_if_not_exist(dst_dir)  # 创建文件夹

    random.seed(47)
    paths_list, names_list = traverse_dir_files(src_dir)
    random.shuffle(paths_list)

    for i, path in enumerate(paths_list):
        if i == num:
            break

        out_path = os.path.join(dst_dir, "{:05d}.{}.jpg".format(i + 1, type))
        shutil.copy(path, out_path)

    print('[Info] 处理完成!')


def main():
    process_s2azhengsheng()


if __name__ == '__main__':
    main()
